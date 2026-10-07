@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Diagnostico Acustico - Instalador e Execucao
cd /d "%~dp0"

set "PASTA=%~dp0"
set "PYEXE=%PASTA%.venv\Scripts\python.exe"
set "MARCA_REQUISITOS=.venv\requisitos_instalados.txt"
set "URL_PYTHON=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
:: O Windows limita caminhos a 260 caracteres. O arquivo mais fundo do .venv
:: tem ~165 e a DLL mais funda ~120 -- e DLLs respeitam esse limite MESMO com
:: os "caminhos longos" habilitados. Por isso a pasta do programa precisa ser curta.
set "LIMITE_CAMINHO=85"
set "LIMITE_CAMINHO_LONGO=130"

echo ==========================================================
echo   Diagnostico Acustico de Falhas Mecanicas
echo   Verificando o ambiente antes de iniciar...
echo ==========================================================
echo.

:: ==========================================================
:: ARQUIVOS DO PROGRAMA (o .zip foi extraido?)
:: ==========================================================
if exist "app.py" if exist "requirements.txt" goto :ARQUIVOS_OK
echo ERRO: os arquivos do programa nao estao nesta pasta.
echo.
echo Se voce abriu este arquivo direto de dentro do .zip: feche esta
echo janela, clique com o botao direito no .zip, escolha "Extrair tudo..."
echo e rode o iniciar.bat de dentro da pasta extraida.
goto :FALHA
:ARQUIVOS_OK

call :VERIFICAR_CAMINHO
if not "%errorlevel%"=="0" goto :FALHA

:: ==========================================================
:: AMBIENTE VIRTUAL (.venv) -- ja existe e funciona?
:: ==========================================================
if not exist "%PYEXE%" goto :PREPARAR_AMBIENTE
"%PYEXE%" -c "import sys" <nul >nul 2>nul
if "%errorlevel%"=="0" goto :AMBIENTE_OK
echo AVISO: o ambiente .venv desta pasta nao funciona neste computador
echo ^(provavelmente foi copiado de outro PC^). Ele sera recriado.
echo.
rmdir /s /q ".venv"

:PREPARAR_AMBIENTE

:: ==========================================================
:: PYTHON
:: ==========================================================
call :PROCURAR_PYTHON
if defined PYBASE goto :PYTHON_OK
echo [ ] Python 3.11 a 3.14 ^(64 bits^) nao encontrado. Instalando o Python 3.12...
echo.
call :INSTALAR_PYTHON
call :PROCURAR_PYTHON
if defined PYBASE goto :PYTHON_OK
echo.
echo ERRO: nao foi possivel instalar o Python automaticamente.
echo Instale o Python 3.12 em https://www.python.org/downloads/
echo marcando "Add python.exe to PATH" na primeira tela do instalador,
echo depois clique duas vezes neste arquivo de novo.
goto :FALHA

:PYTHON_OK
echo [OK] Python encontrado: %PYBASE%
echo.
echo Criando o ambiente do programa ^(.venv^)...
%PYBASE% -m venv ".venv"
if not "%errorlevel%"=="0" goto :FALHA_VENV
if not exist "%PYEXE%" goto :FALHA_VENV
"%PYEXE%" -m pip install --upgrade pip >nul 2>nul

:AMBIENTE_OK
echo [OK] Ambiente do programa pronto.

:: ==========================================================
:: BIBLIOTECAS PYTHON (so reinstala se o requirements.txt mudou)
:: ==========================================================
if not exist "%MARCA_REQUISITOS%" goto :INSTALAR_BIBLIOTECAS
fc /b "requirements.txt" "%MARCA_REQUISITOS%" >nul 2>nul
if "%errorlevel%"=="0" goto :BIBLIOTECAS_OK

:INSTALAR_BIBLIOTECAS
echo.
echo Instalando as bibliotecas Python. Na primeira vez isso baixa
echo varias centenas de MB e pode levar de 5 a 20 minutos -- e normal.
echo.
"%PYEXE%" -m pip install -r requirements.txt
if not "%errorlevel%"=="0" goto :FALHA_BIBLIOTECAS
copy /y "requirements.txt" "%MARCA_REQUISITOS%" >nul

:BIBLIOTECAS_OK
echo [OK] Bibliotecas Python instaladas.

:: ==========================================================
:: MODELO DE LINGUAGEM (spaCy)
:: ==========================================================
"%PYEXE%" -c "import pt_core_news_sm" <nul >nul 2>nul
if "%errorlevel%"=="0" goto :SPACY_OK
echo.
echo Baixando o modelo de linguagem em portugues ^(spaCy^)...
"%PYEXE%" -m spacy download pt_core_news_sm
if not "%errorlevel%"=="0" goto :FALHA_SPACY
:SPACY_OK
echo [OK] Modelo de linguagem instalado.

:: ==========================================================
:: CONFIGURACAO DO STREAMLIT
:: ==========================================================
if not exist ".streamlit" mkdir ".streamlit"
if exist ".streamlit\config.toml" goto :CONFIG_OK
echo Criando configuracao do Streamlit ^(.streamlit\config.toml^)...
(
    echo [server]
    echo maxUploadSize = 10240
    echo maxMessageSize = 10240
    echo address = "localhost"
    echo showEmailPrompt = false
    echo(
    echo [theme]
    echo base = "light"
    echo primaryColor = "#c8102e"
    echo(
    echo [browser]
    echo gatherUsageStats = false
) > ".streamlit\config.toml"
:CONFIG_OK

:: ==========================================================
:: EXECUTAR
:: ==========================================================
echo.
echo ==========================================================
echo   Tudo pronto! Iniciando o Diagnostico Acustico...
echo   O navegador vai abrir sozinho em alguns segundos, no
echo   endereco http://localhost:8501
echo.
echo   Para encerrar o programa, feche esta janela ou aperte Ctrl+C.
echo ==========================================================
echo.
"%PYEXE%" -m streamlit run app.py

echo.
echo O programa foi encerrado.
pause
exit /b 0

:: ==========================================================
:: SAIDAS DE ERRO
:: ==========================================================
:FALHA_VENV
echo.
echo ERRO ao criar o ambiente virtual ^(.venv^).
goto :FALHA

:FALHA_BIBLIOTECAS
echo.
echo ERRO ao instalar as bibliotecas Python. Veja a mensagem em
echo vermelho acima, confira a conexao com a internet e rode de novo.
goto :FALHA

:FALHA_SPACY
echo.
echo ERRO ao baixar o modelo de linguagem. Confira a conexao com a
echo internet e rode de novo.
goto :FALHA

:FALHA
echo.
pause
exit /b 1

:: ==========================================================
:: SUBROTINA -- procura um Python DE VERDADE (3.11 a 3.14, 64 bits).
:: Cada candidato e executado: o atalho "python" da Microsoft Store
:: (que so abre a loja) falha no teste e e ignorado. Tambem procura
:: direto nas pastas de instalacao, para achar um Python recem-
:: instalado sem precisar fechar e abrir esta janela.
:: ==========================================================
:PROCURAR_PYTHON
set "PYBASE="
for %%V in (3.12 3.13 3.11 3.14) do if not defined PYBASE call :TESTAR_PYTHON py -%%V
if not defined PYBASE call :TESTAR_PYTHON python
if not defined PYBASE call :TESTAR_PYTHON python3
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*" "%LOCALAPPDATA%\Python\pythoncore-3*" "%ProgramFiles%\Python3*") do if not defined PYBASE if exist "%%~D\python.exe" call :TESTAR_PYTHON "%%~D\python.exe"
exit /b 0

:TESTAR_PYTHON
%* -c "import sys, struct; sys.exit(0 if sys.version_info[0] == 3 and sys.version_info[1] in range(11, 15) and struct.calcsize('P') == 8 else 1)" <nul >nul 2>nul
:: Compara com zero exatamente: o "py" devolve codigos NEGATIVOS quando a
:: versao nao existe, e "if errorlevel 1" os trataria como sucesso.
if not "%errorlevel%"=="0" exit /b 0
set "PYBASE=%*"
exit /b 0

:: ==========================================================
:: SUBROTINA -- instala o Python 3.12: pelo winget e, se nao der,
:: pelo instalador oficial do python.org (so para este usuario,
:: sem pedir senha de administrador).
:: ==========================================================
:INSTALAR_PYTHON
where winget >nul 2>nul
if not "%errorlevel%"=="0" goto :INSTALAR_PYTHON_DIRETO
echo Instalando o Python 3.12 pelo winget...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :ATUALIZAR_PATH
call :PROCURAR_PYTHON
if defined PYBASE exit /b 0
echo Tentando o winget novamente, sem restringir ao usuario atual...
winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
call :ATUALIZAR_PATH
call :PROCURAR_PYTHON
if defined PYBASE exit /b 0

:INSTALAR_PYTHON_DIRETO
echo Baixando o instalador oficial do Python ^(python.org^)...
set "INSTALADOR=%TEMP%\instalador-python-3.12.10-amd64.exe"
powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri $env:URL_PYTHON -OutFile $env:INSTALADOR"
if exist "%INSTALADOR%" goto :INSTALAR_PYTHON_EXECUTAR
echo Nao foi possivel baixar o instalador do Python.
exit /b 1
:INSTALAR_PYTHON_EXECUTAR
echo Instalando o Python 3.12 ^(apenas para este usuario^)...
start "" /wait "%INSTALADOR%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 InstallLauncherAllUsers=0 Include_test=0
del /q "%INSTALADOR%" >nul 2>nul
call :ATUALIZAR_PATH
exit /b 0

:: ==========================================================
:: SUBROTINA -- uma pasta com caminho comprido faz a instalacao ou o
:: carregamento das bibliotecas falhar no meio, com erros confusos
:: ("nome do arquivo muito grande"). Melhor avisar antes.
:: ==========================================================
:VERIFICAR_CAMINHO
set "LONGPATHS="
for /f "tokens=3" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2^>nul ^| findstr /i "LongPathsEnabled"') do set "LONGPATHS=%%A"
set "LIMITE=%LIMITE_CAMINHO%"
if /i "%LONGPATHS%"=="0x1" set "LIMITE=%LIMITE_CAMINHO_LONGO%"
set "RESTO=%PASTA%"
set /a TAMANHO=0
:VERIFICAR_CAMINHO_CONTAR
if not defined RESTO goto :VERIFICAR_CAMINHO_FIM
set "RESTO=%RESTO:~1%"
set /a TAMANHO+=1
goto :VERIFICAR_CAMINHO_CONTAR
:VERIFICAR_CAMINHO_FIM
if %TAMANHO% LEQ %LIMITE% exit /b 0
echo ERRO: o caminho desta pasta e comprido demais para este Windows
echo ^(%TAMANHO% caracteres; o maximo e %LIMITE%^):
echo   "%PASTA%"
echo.
echo Mova a pasta inteira para um lugar com caminho curto, por exemplo
echo   C:\DiagnosticoAcustico
echo e rode o iniciar.bat de la.
exit /b 1

:: ==========================================================
:: SUBROTINA -- atualiza o PATH desta janela lendo o valor mais
:: recente direto do registro (necessario porque um instalador
:: rodado agora mesmo nao atualiza o PATH de uma janela ja aberta).
:: ==========================================================
:ATUALIZAR_PATH
set "SYSPATH="
set "USERPATH="
for /f "usebackq tokens=2,*" %%A in (`reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul`) do set "SYSPATH=%%B"
for /f "usebackq tokens=2,*" %%A in (`reg query "HKCU\Environment" /v Path 2^>nul`) do set "USERPATH=%%B"
set "PATH=%SYSPATH%;%USERPATH%;%PATH%"
exit /b 0
