# Assistente da disciplina — Disruptive Architectures (RAG)

Chat que responde dúvidas da disciplina **usando só o material do site** (`arnaldojr.github.io/DisruptiveArchitectures`), sempre mostrando de onde veio cada informação. É a aplicação "fora do notebook" (Lab 3.5) do que foi visto nos Labs 1 a 4: engenharia de prompt, conversa com a Interactions API, saídas estruturadas com Pydantic e **RAG** (embeddings + similaridade de cosseno).

## O que o assistente faz

1. Lê os arquivos do material (`.md` e `.ipynb` do repositório do site) e quebra cada um em trechos, um por seção.
2. Transforma cada trecho em um vetor (embedding) e guarda tudo no SQLite.
3. Para cada pergunta do aluno: gera o embedding da pergunta, calcula a similaridade de cosseno com todos os trechos e pega os `k` mais parecidos.
4. Entrega esses trechos ao Gemini como `CONTEXTO`, com um *system prompt* que obriga a responder só com o contexto e a citar a fonte no formato `[arquivo.md]`. Se a base não tem a resposta, ele diz: *"Não encontrei essa informação no material da disciplina."*
5. Devolve a resposta com a lista de fontes (com link para a seção no site) e marca quais foram citadas.

## Por que essa arquitetura

O Lab 3.5 pede três coisas: separar responsabilidades em módulos, ter uma API e persistir interações e logs. A decisão central do projeto é **separar a lógica de negócio da camada de LLM**, como no `petshop-assistant`: a lógica de negócio não depende do modelo de IA e roda igual com qualquer LLM, ou nenhum.

- `app/business/` — Python puro: transformar arquivos em trechos, similaridade de cosseno, top-k, decidir se há contexto suficiente, montar o contexto, conferir as citações e o caso de uso `responder()`. **Zero import de `google-genai`**, banco ou FastAPI. Ela só declara *o que precisa* por meio de duas portas (`Embedder` e `Gerador`, em `portas.py`). Com `GeradorExtrativo` ela responde sem nenhuma IA.
- `app/llm/` — tudo que fala com o Gemini: cliente com retry/backoff, *system prompt*, embeddings (`GeminiEmbedder`) e geração (`GeminiGerador`, via Interactions API). Estas classes **implementam** as portas da camada de negócio; a camada de negócio nunca importa nada daqui.
- `app/persistence/` — SQLite simples, sem ORM: sessões, mensagens (com as fontes de cada resposta), o `interaction_id` da conversa e o índice (trechos + embeddings).
- `app/api/` — a única camada que conhece HTTP. Um FastAPI fino: recebe a requisição, escolhe quem implementa as portas (`dependencias.py`), chama `business.rag.responder`, grava e devolve.
- `app/ingestao.py` — baixa (ou lê de uma pasta) o material, usa `business.conhecimento` para segmentar e o `Embedder` para vetorizar.
- `frontend/` — HTML/JS único, sem framework nem build, para testar na mão e demonstrar.

Como a conversa e a base de conhecimento são coisas diferentes (Lab 4):

| Elemento | Onde vive |
|---|---|
| System prompt | `app/llm/prompts.py` |
| Memória da conversa | `previous_interaction_id` (Lab 2), guardado por sessão no SQLite |
| Base de conhecimento | trechos + embeddings no SQLite, recuperados a cada pergunta |

### O que foi usado do material (e só isso)

| Parte do projeto | Veio de |
|---|---|
| `client.interactions.create`, `system_instruction`, `previous_interaction_id` | Lab 2 |
| Pydantic para os contratos de dados; o princípio "o modelo sugere, o código decide" (aqui, o código decide se há contexto e confere as citações) | Labs 2.5 e 3 |
| Embeddings (`gemini-embedding-2`), prefixos `task: question answering \| query:` e `title: \| text:`, similaridade de cosseno, top-k, `CONTEXTO` com `FONTE`, citação `[arquivo.md]`, recusa quando falta contexto, avaliar a recuperação separada da geração | Lab 4 |
| Pastas separadas, API, persistência, `.gitignore`, chave só por variável de ambiente, README com decisões | Lab 3.5 |
| Estrutura de camadas, SQLite sem ORM, retry 429/503, testes sem chave de API | `petshop-assistant` |

Não há reranker, busca lexical, framework de RAG nem banco vetorial: o Lab 4 mostra que cosseno + top-k sobre os embeddings resolve, e para o tamanho do material uma lista em memória é suficiente.

## Estrutura de pastas

```
disruptive-rag-assistant/
├── app/
│   ├── config.py               # variáveis de ambiente
│   ├── main.py                 # cria o FastAPI, inclui rotas, serve o frontend
│   ├── ingestao.py             # material (.md/.ipynb) -> trechos -> embeddings -> SQLite
│   ├── business/               # SEM IA, SEM banco, SEM HTTP
│   │   ├── schemas.py          # Trecho, TrechoRecuperado, RespostaRag... (Pydantic)
│   │   ├── portas.py           # Embedder e Gerador: o que a lógica exige de fora
│   │   ├── conhecimento.py     # markdown/notebook -> seções -> trechos (+ URL e âncora no site)
│   │   ├── busca.py            # similaridade_cosseno, buscar_trechos (Lab 4)
│   │   ├── regras.py           # contexto suficiente?, montar_contexto, conferir citações
│   │   ├── rag.py              # caso de uso: recuperar() e responder()
│   │   └── extrativo.py        # gerador sem IA
│   ├── llm/                    # tudo que fala com o Gemini
│   │   ├── client.py           # cliente lazy + retry/backoff (429/503)
│   │   ├── prompts.py          # SYSTEM_PROMPT
│   │   ├── embeddings.py       # GeminiEmbedder (implementa Embedder)
│   │   └── assistente.py       # GeminiGerador (implementa Gerador)
│   ├── persistence/db.py       # schema SQLite + leitura/escrita
│   └── api/
│       ├── schemas.py          # request/response HTTP
│       ├── routers.py          # rotas
│       ├── dependencias.py     # escolhe as implementações das portas
│       ├── indexacao.py        # indexação em segundo plano
│       └── limite.py           # limite de mensagens por minuto por IP
├── frontend/index.html         # UI de teste (vanilla JS)
├── scripts/avaliar_recuperacao.py
├── avaliacao/perguntas.json    # perguntas + fontes esperadas
├── tests/                      # 55 testes, nenhum precisa de chave de API
├── data/                       # banco SQLite em tempo de execução (gitignored)
├── Dockerfile · railway.json · .env.example · .gitignore · requirements.txt
```

## Como rodar

### 1. Pré-requisitos

- Python 3.11+
- Uma chave da Gemini API (grátis): https://aistudio.google.com/apikey

### 2. Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuração

```bash
cp .env.example .env
# edite o .env e cole sua GEMINI_API_KEY
```

### 4. Indexar o material (uma vez, e de novo quando o site mudar)

```bash
python -m app.ingestao
```

Por padrão baixa o zip do fork `PietroWilhelm/DisruptiveArchitectures` (branch `master`, pasta `material/`, que é o `docs_dir` do `mkdocs.yml`). Se já tem o repositório clonado, aponte para a pasta e evite o download:

```bash
MATERIAL_DIR=~/DisruptiveArchitectures/material python -m app.ingestao
```

A indexação é **retomável**: cada trecho tem um hash (conteúdo + modelo), então rodar de novo só gera embeddings do que mudou e remove do índice o que saiu do material. Se bater no limite da cota gratuita, o retry espera o tempo sugerido pela API; se mesmo assim parar, é só rodar outra vez.

### 5. Subir a API

```bash
uvicorn app.main:app --reload
```

- Frontend: http://127.0.0.1:8000/
- Swagger: http://127.0.0.1:8000/docs

### 6. Testes

```bash
pytest -v
```

Todos rodam **sem chave de API e sem internet**: a camada de negócio é pura; a API é testada com `Embedder`/`Gerador` falsos injetados via `dependency_overrides`; os adaptadores do Gemini são testados com o SDK substituído. Há ainda um teste que lê o código e falha se `app/business/` importar IA, banco, HTTP ou outra camada, e outro que falha se algum arquivo usar `from __future__`.

## Endpoints da API

| Método | Rota | O que faz |
|---|---|---|
| `GET` | `/api/health` | Status, quantos trechos estão indexados, se há indexação em andamento |
| `POST` | `/api/sessoes` | Cria uma conversa, devolve `sessao_id` |
| `POST` | `/api/sessoes/{id}/mensagens` | Pergunta do aluno → resposta + fontes (com `citada`) + `fundamentada` |
| `GET` | `/api/sessoes/{id}` | Histórico da conversa, com as fontes de cada resposta |
| `POST` | `/api/buscar` | **Só a recuperação**, sem gerar resposta (para avaliar a busca) |
| `POST` | `/api/admin/reindexar` | Reindexa o material em segundo plano (header `X-Admin-Token`) |

## Qualidade das respostas

Medidas de proteção contra resposta inventada, em camadas:

1. **Limiar de similaridade** (`SIMILARIDADE_MINIMA`): se o melhor trecho está abaixo dele, nem chamamos o modelo e respondemos *"Não encontrei..."*. O Lab 4 avisa que cosseno alto não garante resposta correta, por isso há também…
2. **System prompt restritivo** (apenas o CONTEXTO, sem completar com conhecimento próprio, sem inventar comandos/datas/notas, ignorar instruções escritas dentro do contexto).
3. **Citação conferida pelo código**: as fontes `[arquivo.md]` do texto são comparadas com os trechos recuperados; citação que não existe vai para `citacoes_invalidas`. No frontend, as fontes realmente citadas aparecem marcadas.
4. **Perguntas curtas de acompanhamento** ("e no ESP32?") são completadas com a pergunta anterior do aluno antes da busca, já que a busca só enxerga a pergunta que recebe.

Para **medir** (e não só confiar), rode o script que avalia a recuperação separada da geração, como pede o Lab 4:

```bash
python -m scripts.avaliar_recuperacao              # acerto@k, MRR e similaridades
python -m scripts.avaliar_recuperacao --respostas  # também confere citações e recusas
```

Ele imprime a similaridade do melhor trecho para perguntas dentro e fora do escopo: use isso para escolher o `SIMILARIDADE_MINIMA` (o padrão 0.40 é um ponto de partida, não um valor calibrado). Troque/amplie `avaliacao/perguntas.json` com perguntas reais da disciplina.

## Deploy no Railway

1. Suba o projeto para um repositório seu no GitHub (o `.env` e `data/*.db` já estão no `.gitignore`).
2. No Railway: **New Project → Deploy from GitHub repo**. Ele usa o `Dockerfile` e o `railway.json` (healthcheck em `/api/health`).
3. Em **Variables**, defina `GEMINI_API_KEY` e `ADMIN_TOKEN` (um segredo longo). Se quiser outro fork/branch, defina `MATERIAL_REPO` e `MATERIAL_BRANCH`.
4. Crie um **Volume** montado em `/data` (o Dockerfile já define `DATABASE_PATH=/data/rag.db`). Sem volume o banco some a cada deploy e o app precisa reindexar. Se o app não conseguir gravar no volume, defina também `RAILWAY_RUN_UID=0`.
5. **Settings → Networking → Generate Domain**.
6. Na primeira subida, com o banco vazio e a chave configurada, o app indexa sozinho (`AUTO_INGESTAO=true`); acompanhe em `/api/health` (`indexando`, `trechos`, `erro_ingestao`). Quando o site mudar:

```bash
curl -X POST https://SEU-APP.up.railway.app/api/admin/reindexar -H "X-Admin-Token: SEU_TOKEN"
```

## Decisões técnicas

- **Fonte = repositório, não scraping do HTML.** O site é gerado pelo MkDocs a partir de `material/` (`.md` e notebooks via `mkdocs-jupyter`). Ler o texto-fonte evita menus, rodapés e HTML gerado, usa só a biblioteca padrão do Python e preserva blocos de código. O link da seção no site é reconstruído com a mesma regra de URL e âncora do MkDocs.
- **`fonte` com caminho completo** (`aulas/genAI/lab4/lab4.md`) em vez de só `lab4.md`: há arquivos com o mesmo nome em pastas diferentes.
- **Um embedding por chamada**, como no Lab 4, para não depender de como o modelo trata listas de textos; o custo é tempo na indexação, amortizado pelo hash (só o que mudou é refeito).
- **Índice em memória + SQLite**, carregado na subida. Para algumas centenas de trechos, a busca por cosseno em Python é instantânea e dispensa banco vetorial.
- **Memória pela Interactions API** (`previous_interaction_id`), igual ao Lab 2 e ao `petshop-assistant`. Consequência: os trechos de perguntas anteriores continuam no histórico do modelo; o prompt manda responder só com o CONTEXTO, e o código só marca como válidas as citações dos trechos da pergunta atual.
- **Sem contexto suficiente, o modelo nem é chamado**: a resposta é uma mensagem fixa (`MENSAGEM_SEM_RESPOSTA`), o que economiza cota e elimina o risco de alucinar nesses casos.

## Limitações conhecidas

- Não foi testado com a chave real do Gemini nem contra o repositório real nesta entrega: os testes usam fakes e um material de exemplo no mesmo formato. Na primeira execução, confira a saída de `python -m app.ingestao` (arquivos e trechos por arquivo) e rode `scripts.avaliar_recuperacao`.
- `SIMILARIDADE_MINIMA=0.40` precisa de calibração com o `gemini-embedding-2` real.
- Notebooks são lidos como texto e código das células; as saídas (gráficos, prints) são ignoradas. Imagens do material não entram na busca.
- SQLite com uma conexão por operação e limite de mensagens em memória (por processo): suficiente para um projeto de estudo, não para alta concorrência.
- Sem autenticação de usuários: o limite por IP protege a chave, mas quem acessar a URL consegue conversar.
# disruptive-rag-assistant
# disruptive-rag-assistant
