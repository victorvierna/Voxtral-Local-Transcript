# Voxtray

Voxtray es una utilidad local de transcripción en tiempo real (CLI + bandeja del sistema) construida alrededor de `mistralai/Voxtral-Mini-4B-Realtime-2602`.

Está pensada para uso diario en Ubuntu/WSL2 con activación rápida (`toggle`), copiado automático al portapapeles e historial reciente.

## Características

- Grabación por `start/stop/toggle` desde terminal.
- Atajo global en GNOME (`Super+F9` por defecto).
- Copia automática del texto final al portapapeles.
- Historial persistente (últimas 5 transcripciones por defecto).
- Modo bandeja (`tray`) con acciones rápidas y estado del motor.
- Transcripción de archivos de audio (`transcribe-file`).
- Flujo de distribución para Windows + WSL2 (scripts incluidos).

## Requisitos

### Ubuntu 24.04 / WSL2

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg xclip libportaudio2 libxcb-cursor0
```

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

Si quieres modo bandeja Qt:

```bash
pip install -e '.[ui]'
```

Instalar `vLLM` (GPU NVIDIA):

```bash
pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

Acceso a Hugging Face (opcional):
- Para modelos públicos, normalmente no hace falta token.
- Solo necesitas token si el modelo está restringido/gated o es privado.

```bash
export HF_TOKEN=...
```

## Inicio rápido

Inicializar configuración:

```bash
voxtray init
```

Ver estado y logs:

```bash
voxtray status
voxtray logs --target all --lines 200
```

Grabar desde terminal:

```bash
voxtray record --start
voxtray record --stop
# recomendado para atajo global
voxtray record --toggle
```

Control de motor/modelo:

```bash
voxtray warm on
voxtray warm off
voxtray warm status
voxtray model load
voxtray model unload
voxtray model status
```

Historial:

```bash
voxtray history list
voxtray history copy 1
```

Transcribir archivo de audio:

```bash
voxtray transcribe-file /ruta/audio.m4a --copy
```

## Configuración y perfiles

Archivo principal:

- `~/.config/voxtray/config.toml`

Perfiles de memoria incluidos:

- `profiles/voxtray-balanced.toml`
- `profiles/voxtray-vram-saver.toml`
- `profiles/voxtray-latency.toml`

Aplicar perfil:

```bash
scripts/apply_profile.sh balanced
```

Valores disponibles:

- `balanced`
- `vram-saver`
- `latency`

## Integración GNOME

Instala desktop entry + autostart + shortcut por defecto (`Super+F9`):

```bash
scripts/install_ubuntu_integration.sh
```

Solo instalar shortcut:

```bash
voxtray install-shortcut --binding '<Super>F9'
```

## Windows + WSL2

Generar bundle para compartir:

```bash
scripts/build_wsl2_bundle.sh
```

Instalación inicial en Windows (PowerShell), dentro de la carpeta extraída:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_windows_shortcuts.ps1 -Distro Ubuntu
```

Qué hace este instalador:
- Instala dependencias en WSL2 (Ubuntu).
- Crea el entorno Python y Voxtray en `~/.voxtray`.
- Genera accesos directos de Windows para usar Voxtray sin abrir terminal Linux.

Activación y uso diario en Windows:
1. Usa el acceso directo `Voxtray Toggle` (escritorio o menú inicio) para iniciar/parar grabación.
2. Usa `Voxtray Warm On/Off` para mantener o liberar el motor en memoria.
3. Usa `Voxtray Status` para revisar estado rápido.
4. Usa `Voxtray Logs` si quieres diagnóstico.

Atajo global en Windows (opcional):
1. Haz clic derecho sobre `Voxtray Toggle` y abre `Propiedades`.
2. En la pestaña `Acceso directo`, coloca el cursor en `Tecla de método abreviado`.
3. Pulsa la combinación deseada (recomendado: `Ctrl + Alt + F9`).
4. Pulsa `Aplicar` y `Aceptar`.

Notas:
- El acceso directo también está en `%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Voxtray WSL`.
- Si eliges una tecla simple, Windows suele convertirla a `Ctrl + Alt + <tecla>`.
- Evita combinaciones ya usadas por otras apps o por el sistema.

Verificación rápida desde PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc "~/.voxtray/.venv/bin/voxtray status"
wsl -d Ubuntu -- bash -lc "~/.voxtray/.venv/bin/voxtray warm on"
```

## Desarrollo

Ejecutar tests:

```bash
pytest
```

Estructura del proyecto:

- `src/voxtray/`: implementación principal (CLI, control, realtime, tray).
- `tests/`: pruebas unitarias.
- `scripts/`: integración Ubuntu/WSL2 y utilidades de empaquetado.
- `profiles/`: plantillas de configuración de rendimiento/memoria.

## Seguridad y datos locales

- No subas credenciales ni tokens al repositorio.
- Usa variables de entorno locales (ejemplo en `.env.example`).
- Estado, logs e historial se guardan en tu home:
  - `~/.local/state/voxtray/`
  - `~/.local/share/voxtray/`

## Licencia

Apache-2.0. Ver `LICENSE`.
