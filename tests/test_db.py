import numpy as np

from app.business.schemas import Trecho


def _trecho(n=1):
    return Trecho(fonte=f"a{n}.md", titulo="T", secao=f"T > S{n}", url=f"http://x/{n}", conteudo=f"conteudo {n}")


def test_sessao_mensagens_e_estado(banco):
    sessao = banco.criar_sessao()
    assert banco.sessao_existe(sessao) and not banco.sessao_existe("nao-existe")
    assert banco.obter_ultimo_estado(sessao) is None

    banco.registrar_mensagem(sessao, "aluno", "pergunta 1")
    banco.registrar_mensagem(sessao, "assistente", "resposta 1", [{"fonte": "a.md"}])
    banco.registrar_mensagem(sessao, "aluno", "pergunta 2")
    banco.atualizar_ultimo_estado(sessao, "interaction-9")

    assert banco.obter_ultimo_estado(sessao) == "interaction-9"
    assert banco.listar_perguntas_do_aluno(sessao, limite=2) == ["pergunta 1", "pergunta 2"]
    historico = banco.obter_historico(sessao)
    assert [m["remetente"] for m in historico] == ["aluno", "assistente", "aluno"]
    assert historico[1]["fontes"] == [{"fonte": "a.md"}]


def test_indice_ida_e_volta_do_embedding(banco):
    vetor = np.array([0.1, -0.5, 2.0], dtype=float)
    banco.salvar_trecho("h1", _trecho(1), vetor)
    banco.salvar_trecho("h2", _trecho(2), vetor * 2)

    indice = banco.carregar_indice()
    assert banco.contar_trechos() == 2 and len(indice) == 2
    assert indice[0].trecho == _trecho(1)
    assert np.allclose(indice[0].embedding, vetor, atol=1e-6)


def test_remover_trechos_obsoletos(banco):
    banco.salvar_trecho("h1", _trecho(1), np.ones(2))
    banco.salvar_trecho("h2", _trecho(2), np.ones(2))
    assert banco.remover_trechos_exceto({"h1"}) == 1
    assert banco.hashes_dos_trechos() == {"h1"}


def test_metadados(banco):
    assert banco.obter_metadado("indexado_em") is None
    banco.definir_metadado("indexado_em", "2026-10-06")
    assert banco.obter_metadado("indexado_em") == "2026-10-06"


def test_exportar_e_importar_indice_nao_leva_sessoes(banco, tmp_path, monkeypatch):
    import numpy as np

    from app.business.schemas import Trecho

    t = Trecho(fonte="a.md", titulo="A", secao="A", url="u", conteudo="conteúdo de teste do trecho")
    banco.salvar_trecho("h1", t, np.array([1.0, 2.0]))
    banco.definir_metadado("indexado_em", "hoje")
    banco.criar_sessao()

    semente = tmp_path / "seed" / "indice.db"
    assert banco.exportar_indice(str(semente)) == 1

    from app import config

    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "novo.db"))    # banco vazio, como num deploy novo
    banco.inicializar_schema()
    assert banco.contar_trechos() == 0
    assert banco.importar_indice(str(semente)) == 1
    assert banco.importar_indice(str(semente)) == 0         # de novo: nada duplicado
    item = banco.carregar_indice()[0]
    assert item.trecho.fonte == "a.md" and np.allclose(item.embedding, [1.0, 2.0])
    assert banco.obter_metadado("indexado_em") == "hoje"
