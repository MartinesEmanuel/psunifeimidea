# Diagnóstico Acústico de Falhas Mecânicas

Aplicação local para apoiar a identificação de falhas mecânicas a partir de **áudio de vídeos ou gravações** e de um **laudo em texto**. O sistema extrai a trilha sonora, identifica e remove voz humana quando necessário, combina atributos acústicos e textuais e estima a probabilidade dos problemas já conhecidos.

> O resultado é um apoio à decisão. Toda indicação deve ser confirmada por inspeção técnica do equipamento.

## O que o sistema faz

- **Alimentação (Admin):** cadastra casos com problema conhecido, áudio/vídeo, laudo e rótulo.
- **Diagnóstico (Operador):** analisa um caso desconhecido e apresenta as probabilidades por problema.
- **Aprendizado supervisionado:** ao confirmar ou corrigir um diagnóstico, o caso pode entrar na base e o modelo é retreinado.
- **Processamento local:** vídeos, áudios, laudos e modelo ficam no computador; a interface padrão atende somente em `localhost`.
- **Backup:** permite baixar a base, os áudios, os laudos e o modelo treinado em um único arquivo `.zip`.

## Versão principal

A `main` contém a versão final aprovada do **app midea completo**, diretamente na raiz do repositório. O código, o instalador, o APK, os guias, a base de exemplos, o modelo treinado e os pacotes para celulares vêm dessa versão. Os arquivos históricos ficam em `_versoes_anteriores/`.

Execute `iniciar.bat` no Windows ou siga a execução manual abaixo no Linux/macOS. O `app.py` da raiz é a entrada da versão atual.

Configure a conexão em **Alimentação (Admin) → Celulares e sincronização**. Marque **Este computador é a Central** somente no PC responsável pelos celulares; os demais PCs trocam a base sem essa opção. A preparação do Supabase está em [`nuvem/COMO_CONFIGURAR.txt`](nuvem/COMO_CONFIGURAR.txt). Senhas e chaves são locais e não entram no Git.

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

A base de exemplos e o modelo da versão aprovada estão incluídos neste repositório. Novos arquivos gerados pelo uso continuam ignorados pelo Git e ficam ao lado de `app.py`:

```text
dataset_master.csv       # casos, metadados e atributos calculados
audios_conhecidos/       # WAVs processados
textos_conhecidos/       # laudos normalizados
modelos/                 # LightGBM, codificador e metadados do treino
```

Para migrar a base, use **Baixar backup (.zip)** na barra lateral ou copie esses quatro itens para a pasta do projeto no computador de destino. Não copie `.venv/` entre máquinas.

## Documentação

- [Documentação técnica histórica](docs/ARQUITETURA_TECNICA.md) — referência da arquitetura anterior; a versão atual está dividida em `diagnostico_acustico/`.
- [Guia técnico de instalação e operação](Guia_Tecnico_Diagnostico_Acustico.pdf) — manual para TI, administradores e operadores.
- [Instalação no celular](app_celular/COMO_INSTALAR_NO_CELULAR.txt) — instalação do APK.

## Estrutura do repositório

```text
app.py                         # entrada Streamlit da versão final
diagnostico_acustico/           # pipeline de ML, sincronização e telas
estilo/                        # identidade visual
app_celular/                   # APK e instruções
nuvem/                         # configuração do Supabase
iniciar.bat                    # instalador/inicializador para Windows
requirements.txt               # dependências Python
.streamlit/config.toml         # configuração local do Streamlit
docs/ARQUITETURA_TECNICA.md    # documentação técnica
```

## Segurança

A configuração padrão usa `address = "localhost"`, sem telemetria. O modo administrador exige uma senha local, criada no primeiro acesso. Senhas e chaves da sincronização não são versionadas. Mantenha o acesso local, salvo se houver uma configuração de rede e controles de acesso adequados.
