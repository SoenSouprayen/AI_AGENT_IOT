@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title AI_AGENT_IOT - verification et lancement

set "MODEL=llama3.2:3b"
set "DRIVE_URL=https://drive.google.com/file/d/15GaVgGYw6PLx6f8GllSbyRHXPeT2gdlq/view?usp=drive_link"
set "DRIVE_ID=15GaVgGYw6PLx6f8GllSbyRHXPeT2gdlq"
set "RAW_NAME=packet_level_dataset_with_geometry_doppler_rp002_1_0_5.csv"
set "PROC_NAME=scenario_dataset_1deg.csv"
set "RAW_FILE=data\raw\%RAW_NAME%"
set "PROC_FILE=data\processed\%PROC_NAME%"
set "MODEL_FILE=models\per_model\per_predictor_A_Geometry_1deg.joblib"

echo ============================================================
echo  AI_AGENT_IOT - verification de l'environnement
echo ============================================================

rem ---------- 1. Python ----------
echo.
echo [1/6] Python...
set "PY="
where py >nul 2>&1 && set "PY=py"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY goto :err_python
echo     OK : %PY%

rem ---------- 2. Environnement virtuel ----------
echo.
echo [2/6] Environnement virtuel .venv...
if exist ".venv\Scripts\python.exe" goto :venv_ok
if exist ".venv" (
    echo     .venv present mais inutilisable sous Windows, recreation...
    rmdir /s /q ".venv"
)
echo     Creation du .venv...
%PY% -m venv .venv
if errorlevel 1 goto :err_venv
:venv_ok
call ".venv\Scripts\activate.bat"
echo     OK : .venv active

rem ---------- 3. Bibliotheques ----------
echo.
echo [3/6] Bibliotheques Python (requirements.txt)...
if not exist "requirements.txt" goto :err_req
python -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :err_pip
echo     OK : bibliotheques a jour

rem ---------- 4. Donnees ----------
echo.
echo [4/6] CSV brut...
if exist "%RAW_FILE%" goto :data_ok
echo     Fichier(s) manquant(s).
if exist "_download" rmdir /s /q "_download"
mkdir "_download"
if exist "data.zip" (
    echo     data.zip trouve dans le dossier du projet, utilisation de cette archive.
    copy /y "data.zip" "_download\data.zip" >nul
    goto :unzip
)
echo     Telechargement depuis Google Drive...
python -m pip install -q --disable-pip-version-check gdown
if errorlevel 1 goto :err_gdown
python -m gdown "%DRIVE_ID%" -O "_download\data.zip"
if errorlevel 1 goto :manual_dl
if exist "_download\data.zip" goto :unzip
:manual_dl
echo.
echo     Le telechargement automatique a echoue : le lien Drive n'est probablement pas
echo     partage en "Toute personne disposant du lien", ou le quota est depasse.
echo     Ouverture du lien dans ton navigateur : telecharge le zip, puis reviens ici.
start "" "https://drive.google.com/uc?export=download&id=%DRIVE_ID%"
echo.
pause
powershell -NoProfile -Command "$c=@('.',(Join-Path $env:USERPROFILE 'Downloads')) | ForEach-Object { Get-ChildItem $_ -Filter *.zip -File -ErrorAction SilentlyContinue } | Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-2) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($c){ Copy-Item $c.FullName '_download\data.zip' -Force; Write-Host ('    archive utilisee : '+$c.FullName); exit 0 } else { exit 1 }"
if errorlevel 1 goto :err_dl
:unzip
echo     Extraction de l'archive...
powershell -NoProfile -Command "Expand-Archive -LiteralPath '_download\data.zip' -DestinationPath '_download\extracted' -Force"
if errorlevel 1 goto :err_unzip
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
powershell -NoProfile -Command "$f=Get-ChildItem '_download\extracted' -Recurse -File -Filter '%RAW_NAME%' | Select-Object -First 1; if($f){ Copy-Item $f.FullName '%RAW_FILE%' -Force; Write-Host '    copie : %RAW_FILE%' }"
if exist "%RAW_FILE%" (
    rmdir /s /q "_download"
    goto :data_ok
)
goto :err_data
:data_ok
echo     OK : CSV brut present

rem ---------- 5. Fichiers generes ----------
echo.
echo [5/6] Fichiers generes ^(dataset traite et modele^)...
set "DATASET_PATH=%~dp0data\raw\%RAW_NAME%"
if exist "%PROC_FILE%" goto :proc_ok
echo     Generation de %PROC_NAME% depuis le CSV brut, cela peut prendre quelques minutes...
python scripts\Database\prepare_dataset.py
if errorlevel 1 goto :err_prepare
if not exist "%PROC_FILE%" goto :err_prepare
:proc_ok
echo     OK : %PROC_NAME% present
if exist "%MODEL_FILE%" goto :model_ok
echo     Entrainement du modele PER, cela peut prendre quelques minutes...
python scripts\ML\train_geometry_per_model.py --dataset "%PROC_FILE%" --model "%MODEL_FILE%"
if errorlevel 1 goto :err_train
if not exist "%MODEL_FILE%" goto :err_train
:model_ok
echo     OK : modele present

rem ---------- Lancement de l'application web ----------
:launch_web
echo.
echo ============================================================
echo  Tout est pret. Lancement de l'application web...
echo ============================================================
echo.
<<<<<<< HEAD
choice /c 123456 /n /m "Ton choix : "
set "CH=%errorlevel%"
if "%CH%"=="6" goto :end
if "%CH%"=="5" goto :run_tests
if "%CH%"=="1" goto :run_nollm
if "%LLM_OK%"=="0" (
    echo.
    echo Mode LLM indisponible ^(voir etape 6^). Choisis une autre option.
    goto :menu
)
if "%CH%"=="2" goto :run_llm1
if "%CH%"=="3" goto :run_llm2
if "%CH%"=="4" goto :run_inter
goto :menu

:run_nollm
python app.py --no-llm --N 10000
goto :after_run
:run_llm1
python app.py --llm-model %MODEL% --N 10000 --require-llm --use-rag --question "Explique simplement pourquoi ce mode est recommande."
goto :after_run
:run_llm2
python app.py --llm-model %MODEL% --N 100000 --require-llm --use-rag --question "Compare les PER de DR8 et DR9 et explique le choix."
goto :after_run
:run_inter
python app.py --llm-model %MODEL% --interactive --use-rag
goto :after_run
:run_tests
python -m unittest discover -s tests -p "test_*.py" -v
goto :after_run

:after_run
=======
echo  URL ^(cliquable dans Windows Terminal^) :
echo     http://localhost:8501
echo  ^(si 8501 est occupe, l'URL finale sera affichee ci-dessous^)
>>>>>>> 782088f0c3ccf54d38632e90538baa7a967aef96
echo.
echo  Appuyez sur Ctrl+C pour arreter le serveur web.
echo.

"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\run_local.py" %*

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Interface web arretee ^(code %EXIT_CODE%^).
pause
endlocal & exit /b %EXIT_CODE%

rem ---------- Erreurs ----------
:err_python
echo     ERREUR : Python introuvable. Installe-le depuis https://www.python.org ^(coche "Add to PATH"^).
goto :fail
:err_venv
echo     ERREUR : creation du .venv impossible.
goto :fail
:err_req
echo     ERREUR : requirements.txt introuvable. Lance ce script depuis la racine du projet.
goto :fail
:err_pip
echo     ERREUR : installation des bibliotheques echouee.
goto :fail
:err_gdown
echo     ERREUR : installation de gdown echouee.
goto :fail
:err_dl
echo     ERREUR : aucun zip trouve. Telecharge-le a la main, place-le dans le dossier du projet sous le nom data.zip, puis relance ce script.
goto :fail
:err_unzip
echo     ERREUR : extraction du zip echouee.
goto :fail
:err_data
echo     ERREUR : CSV brut introuvable dans l'archive. Contenu extrait dans _download\extracted :
dir /s /b "_download\extracted"
goto :fail
:err_prepare
echo     ERREUR : la generation de %PROC_NAME% a echoue ^(voir le message ci-dessus^).
goto :fail
:err_train
echo     ERREUR : l'entrainement du modele a echoue, ou %MODEL_FILE% n'a pas ete cree ^(voir ci-dessus^).
goto :fail

:fail
echo.
pause
endlocal & exit /b 1