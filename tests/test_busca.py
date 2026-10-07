import numpy as np

from app.business.busca import buscar_trechos, similaridade_cosseno
from app.business.schemas import ItemIndice, Trecho


def _item(nome, vetor):
    return ItemIndice(Trecho(fonte=nome, titulo=nome, secao=nome, url="u", conteudo=nome), np.array(vetor, dtype=float))


def test_cosseno_valores_conhecidos():
    assert similaridade_cosseno(np.array([1, 0]), np.array([1, 0])) == 1.0
    assert abs(similaridade_cosseno(np.array([1, 0]), np.array([0, 1]))) < 1e-9
    assert similaridade_cosseno(np.array([1, 0]), np.array([-1, 0])) == -1.0


def test_cosseno_com_vetor_zero_nao_quebra():
    assert similaridade_cosseno(np.zeros(3), np.array([1, 2, 3])) == 0.0


def test_buscar_ordena_e_limita_em_k():
    indice = [_item("longe", [0, 1]), _item("perto", [1, 0.1]), _item("meio", [1, 1])]
    achados = buscar_trechos(np.array([1.0, 0.0]), indice, k=2)
    assert [a.trecho.fonte for a in achados] == ["perto", "meio"]
    assert achados[0].similaridade > achados[1].similaridade


def test_buscar_em_indice_vazio_e_k_zero():
    assert buscar_trechos(np.array([1.0]), [], k=3) == []
    assert buscar_trechos(np.array([1.0]), [_item("a", [1.0])], k=0) == []
