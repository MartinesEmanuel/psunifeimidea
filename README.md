# Diagnóstico Acústico de Falhas Mecânicas

Aplicação local para apoiar a identificação de falhas mecânicas a partir de **áudio de vídeos ou gravações** e de um **laudo em texto**. O sistema extrai a trilha sonora, identifica e remove voz humana quando necessário, combina atributos acústicos e textuais e estima a probabilidade dos problemas já conhecidos.

> O resultado é um apoio à decisão. Toda indicação deve ser confirmada por inspeção técnica do equipamento.

## O que o sistema faz

- **Alimentação (Admin):** cadastra casos com problema conhecido, áudio/vídeo, laudo e rótulo.
- **Diagnóstico (Operador):** analisa um caso desconhecido e apresenta as probabilidades por problema.
- **Aprendizado supervisionado:** ao confirmar ou corrigir um diagnóstico, o caso pode entrar na base e o modelo é retreinado.
- **Processamento local:** vídeos, áudios, laudos e modelo ficam no computador; a interface padrão atende somente em `localhost`.
- **Backup:** permite baixar a base, os áudios, os laudos e o modelo treinado em um único arquivo `.zip`.

## Início rápido — Windows

1. Baixe ou clone o projeto e mantenha-o em uma pasta local de caminho curto, por exemplo `C:\DiagnosticoAcustico`.
2. Não execute o projeto de dentro de um arquivo `.zip`; extraia-o antes.
3. Execute `iniciar.bat` com a conta que utilizará o sistema. **Não use “Executar como administrador”.**
4. Na primeira execução, aguarde a criação do ambiente Python, a instalação das dependências e o download do modelo de linguagem.
5. Acesse o endereço mostrado no terminal — normalmente [http://localhost:8501](http://localhost:8501).

O computador precisa de Windows 10 (1809+) ou Windows 11, arquitetura x64, ao menos 8 GB de RAM e 5 GB livres (16 GB e SSD são recomendados). A primeira instalação requer internet; depois do download dos modelos, o uso é local.

## Execução manual — Linux, macOS ou ambiente de desenvolvimento

É necessário ter Python 3.11–3.14 e `ffmpeg` disponível no sistema.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download pt_core_news_sm
.venv/bin/python -m streamlit run app.py
```

No Windows, os comandos equivalentes usam `.venv\Scripts\python.exe`.

## Como preparar a base

1. Abra **Alimentação (Admin)**.
2. Envie um ou mais arquivos do mesmo problema, preencha o laudo e informe o rótulo.
3. Para o primeiro treinamento são necessários **pelo menos dois rótulos**, cada um com **10 casos**.
4. Revise as informações e use **Alimentar Sistema**. Ao final do lote, o sistema retreina o modelo quando houver dados suficientes.

Arquivos aceitos:

| Tipo | Extensões |
| --- | --- |
| Vídeo | `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`, `.webm`, `.3gp`, `.wmv` |
| Áudio | `.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac` |

Recomenda-se gravar entre 30 segundos e 3 minutos, próximo ao defeito, evitando conversas, vento e ruídos externos. O limite de upload configurado é 10 GB por arquivo, mas arquivos de até cerca de 500 MB são mais adequados ao processamento em memória.

## Dados, backup e privacidade

Os dados gerados pelo uso não são versionados pelo Git e ficam ao lado de `app.py`:

```text
dataset_master.csv       # casos, metadados e atributos calculados
audios_conhecidos/       # WAVs processados
textos_conhecidos/       # laudos normalizados
modelos/                 # LightGBM, codificador e metadados do treino
```

Para migrar a base, use **Baixar backup (.zip)** na barra lateral ou copie esses quatro itens para a pasta do projeto no computador de destino. Não copie `.venv/` entre máquinas.

## Acesso administrativo

O modo **Alimentação (Admin)** é protegido por senha. Antes de iniciar o aplicativo, copie `.streamlit/secrets.example.toml` para `.streamlit/secrets.toml` e substitua o valor de `admin_password` por uma senha forte. Esse arquivo é ignorado pelo Git e deve ficar somente no computador/servidor autorizado.

Como alternativa de implantação, defina a variável de ambiente `DIAGNOSTICO_ADMIN_PASSWORD`. Técnicos sem essa senha podem apenas consultar o modo **Diagnóstico (Operador)**; o cadastro, exclusão e treinamento ficam restritos à área administrativa.

## Documentação

- [Documentação técnica](docs/ARQUITETURA_TECNICA.md) — arquitetura, pipeline, persistência, modelo e operação para desenvolvimento.
- [Guia técnico de instalação e operação](Guia_Tecnico_Diagnostico_Acustico_Imagens_por_Secao-1.pdf) — manual para TI, administradores e operadores.
- [Instruções resumidas](LEIA-ME.txt) — instalação e uso básico.

## Estrutura do repositório

```text
app.py                         # aplicação Streamlit e pipeline de ML
iniciar.bat                    # instalador/inicializador para Windows
requirements.txt               # dependências Python
.streamlit/config.toml         # configuração local do Streamlit
docs/ARQUITETURA_TECNICA.md    # documentação técnica
```

## Segurança

A configuração padrão usa `address = "localhost"`, sem telemetria. Não altere para acesso de rede sem adicionar autenticação e controles de acesso: a aba administrativa permite incluir e excluir dados.
