import zipfile

import pytest

from app import config, ingestao
from app.llm.client import CotaEsgotadaError


def test_ingestao_de_pasta_local_md_e_ipynb(ia, banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    resumo = ingestao.executar(progresso=lambda *_: None)

    assert resumo["arquivos"] == 3                      # .png ignorado
    assert resumo["novos"] == resumo["trechos"] == banco.contar_trechos() > 3
    fontes = {i.trecho.fonte for i in banco.carregar_indice()}
    assert fontes == {"aulas/genAI/lab4/lab4.md", "aulas/genAI/lab2/lab2.md", "aulas/checkpoint/cp4.ipynb"}
    assert banco.obter_metadado("indexado_em")


def test_segunda_ingestao_nao_reembute_e_remove_o_que_sumiu(ia, banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    ingestao.executar(progresso=lambda *_: None)
    chamadas = ia.chamadas_documento

    resumo = ingestao.executar(progresso=lambda *_: None)
    assert resumo["novos"] == 0 and ia.chamadas_documento == chamadas   # nada novo, nada embutido

    (material / "aulas" / "genAI" / "lab2" / "lab2.md").unlink()
    resumo = ingestao.executar(progresso=lambda *_: None)
    assert resumo["removidos"] >= 1
    assert "aulas/genAI/lab2/lab2.md" not in {i.trecho.fonte for i in banco.carregar_indice()}


def test_conteudo_alterado_gera_trecho_novo(ia, banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    ingestao.executar(progresso=lambda *_: None)
    arquivo = material / "aulas" / "genAI" / "lab2" / "lab2.md"
    arquivo.write_text(arquivo.read_text(encoding="utf-8") + "\nNovo parágrafo sobre sessões.\n", encoding="utf-8")
    resumo = ingestao.executar(progresso=lambda *_: None)
    assert resumo["novos"] >= 1 and resumo["removidos"] >= 1


def test_download_do_zip_do_repositorio(ia, banco, material, tmp_path, monkeypatch):
    """Simula o zip do GitHub: uma pasta raiz 'Repo-master/' com 'material/' dentro."""
    zip_path = tmp_path / "repo.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for arquivo in material.rglob("*"):
            if arquivo.is_file():
                z.write(arquivo, f"Repo-master/material/{arquivo.relative_to(material).as_posix()}")
    monkeypatch.setattr(config, "MATERIAL_DIR", "")
    monkeypatch.setattr(config, "MATERIAL_ZIP_URL", zip_path.as_uri())

    resumo = ingestao.executar(progresso=lambda *_: None)
    assert resumo["arquivos"] == 3


def test_material_vazio_levanta_erro(ia, banco, tmp_path, monkeypatch):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    monkeypatch.setattr(config, "MATERIAL_DIR", str(vazio))
    with pytest.raises(RuntimeError):
        ingestao.executar(progresso=lambda *_: None)


def test_pasta_inexistente(ia, banco, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", "/nao/existe")
    with pytest.raises(FileNotFoundError):
        ingestao.executar(progresso=lambda *_: None)


def test_filtro_incluir_e_excluir(ia, banco, material, monkeypatch):
    (material / "aulas" / "genAI" / "lab4" / "lab4 copy.md").write_text("# Lab 4 copia\n\nconteúdo duplicado da cópia", encoding="utf-8")
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    monkeypatch.setattr(config, "MATERIAL_INCLUIR", ["aulas/genAI"])
    monkeypatch.setattr(config, "MATERIAL_EXCLUIR", ["*copy.*", "*lab2*"])

    resumo = ingestao.executar(progresso=lambda *_: None)
    fontes = {i.trecho.fonte for i in banco.carregar_indice()}
    assert fontes == {"aulas/genAI/lab4/lab4.md"}            # checkpoint fora do include; lab2 e a cópia excluídos
    assert resumo["arquivos"] == 1 and resumo["arquivos_ignorados_pelo_filtro"] == 3


def test_cota_esgotada_no_meio_guarda_o_progresso_e_nao_apaga_nada(ia, banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    ingestao.executar(progresso=lambda *_: None)
    indexados_antes = banco.contar_trechos()

    # o material muda (trechos novos) e a cota acaba depois de 2 embeddings
    arquivo = material / "aulas" / "genAI" / "lab2" / "lab2.md"
    arquivo.write_text(arquivo.read_text(encoding="utf-8") + "\n## A\n\nTexto novo da seção A sobre sessões.\n\n## B\n\nTexto novo da seção B sobre sessões.\n\n## C\n\nTexto novo da seção C sobre sessões.\n", encoding="utf-8")

    ia.falhar_apos, ia.erro = ia.chamadas_documento + 2, CotaEsgotadaError("acabou")
    mensagens = []
    with pytest.raises(CotaEsgotadaError):
        ingestao.executar(progresso=mensagens.append)
    ia.falhar_apos = None

    assert banco.contar_trechos() >= indexados_antes + 2       # os 2 novos ficaram salvos
    assert any("Cota da API esgotada" in m for m in mensagens)

    # rodar de novo (cota voltou) termina o serviço
    resumo = ingestao.executar(progresso=lambda *_: None)
    assert resumo["trechos"] == banco.contar_trechos()


def test_contar_pendentes_nao_chama_a_api_e_respeita_o_cache(ia, banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    antes = ingestao.contar_pendentes(progresso=lambda *_: None)
    assert antes["trechos"] > 3 and antes["faltam"] == antes["trechos"]
    assert ia.chamadas_documento == 0
    assert banco.contar_trechos() == 0                      # contar não grava nada

    ingestao.executar(progresso=lambda *_: None)
    depois = ingestao.contar_pendentes(progresso=lambda *_: None)
    assert depois["faltam"] == 0 and depois["trechos"] == antes["trechos"]
    assert set(depois["por_pasta"]) == {"aulas/genAI/lab4", "aulas/genAI/lab2", "aulas/checkpoint"}


def test_rodada_filtrada_nao_apaga_o_que_esta_fora_do_filtro(ia, banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    ingestao.executar(progresso=lambda *_: None)
    total = banco.contar_trechos()

    monkeypatch.setattr(config, "MATERIAL_INCLUIR", ["aulas/checkpoint"])
    resumo = ingestao.executar(progresso=lambda *_: None)
    assert resumo["removidos"] == 0
    assert banco.contar_trechos() == total                       # genAI continua indexado

    # dentro do filtro, o que sumiu do material ainda é limpo
    (material / "aulas" / "checkpoint" / "cp4.ipynb").unlink()
    (material / "aulas" / "checkpoint" / "cp5.md").write_text("# CP5\n\nConteudo novo do checkpoint cinco.\n" * 3, encoding="utf-8")
    resumo = ingestao.executar(progresso=lambda *_: None)
    assert resumo["removidos"] >= 1
    fontes = {i.trecho.fonte for i in banco.carregar_indice()}
    assert "aulas/checkpoint/cp4.ipynb" not in fontes and "aulas/genAI/lab4/lab4.md" in fontes
