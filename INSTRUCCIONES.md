# Palladium — Predictor de Cancelaciones

Aplicación web para predecir la probabilidad de cancelación de reservas hoteleras.  
Se abre en el navegador; solo necesitas Python instalado.

---

## Opción A — Ejecutar en tu propio ordenador

### 1. Instalar Python (si no lo tienes)

- **Mac / Linux**: descarga desde https://www.python.org/downloads/ e instala normalmente.
- **Windows**: descarga desde https://www.python.org/downloads/ e instala.  
  ⚠️ Durante la instalación marca la casilla **"Add Python to PATH"** antes de darle a Install.

Para comprobar que está instalado, abre una terminal y escribe:
```
python --version
```
Tiene que salir algo como `Python 3.11.x` (cualquier versión 3.10 o superior vale).

---

### 2. Descomprimir el proyecto

Descomprime el ZIP en cualquier carpeta de tu ordenador, por ejemplo el Escritorio.  
Dentro verás una carpeta llamada `Visual Palladium`.

---

### 3. Abrir una terminal en esa carpeta

**Mac:**
1. Abre la app **Terminal** (búscala en Spotlight con Cmd+Espacio).
2. Escribe `cd ` (con espacio al final), arrastra la carpeta `Visual Palladium` a la terminal y pulsa Enter.

**Windows:**
1. Abre la carpeta `Visual Palladium` en el Explorador de archivos.
2. Haz clic en la barra de direcciones, escribe `cmd` y pulsa Enter.

---

### 4. Crear el entorno virtual e instalar dependencias

Copia y pega estos comandos **uno a uno**:

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

El último paso descarga las librerías — puede tardar 1-2 minutos la primera vez.

---

### 5. Arrancar la aplicación

```bash
python app.py
```

Cuando veas este mensaje en la terminal, ya está funcionando:
```
Running on http://127.0.0.1:5050
```

Abre tu navegador y ve a: **http://localhost:5050**

---

### Credenciales de acceso

| Usuario   | Contraseña     |
|-----------|----------------|
| diego     | diego123       |
| erik      | erik123        |
| guillermo | guillermo123   |
| raul      | raul123        |

---

### Para parar y volver a arrancar

- Para parar: pulsa **Ctrl+C** en la terminal.
- La próxima vez solo tienes que activar el entorno y arrancar:

  **Mac:** `source venv/bin/activate && python app.py`  
  **Windows:** `venv\Scripts\activate` y luego `python app.py`

---

## Opción B — Publicar online con Railway (acceso desde cualquier sitio)

Railway es una plataforma gratuita que publica tu app en una URL pública.  
Solo necesitas una cuenta de GitHub.

### Paso 1 — Sube el proyecto a GitHub

1. Ve a https://github.com/new y crea un repositorio nuevo (puede ser privado).
2. En tu terminal, dentro de la carpeta `Visual Palladium`:

```bash
git init
git add .
git commit -m "primera versión"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

> Sustituye `TU_USUARIO` y `TU_REPO` por los tuyos.

---

### Paso 2 — Crea la app en Railway

1. Ve a https://railway.app y entra con tu cuenta de GitHub.
2. Haz clic en **New Project → Deploy from GitHub repo**.
3. Selecciona el repositorio que acabas de crear.
4. Railway detectará automáticamente que es una app Python.

---

### Paso 3 — Configura el comando de inicio

En el panel de Railway, ve a **Settings → Deploy → Start Command** y pon:

```
python app.py
```

---

### Paso 4 — Variables de entorno

En Railway ve a **Variables** y añade:

| Variable | Valor |
|----------|-------|
| `PORT`   | `5050` |

---

### Paso 5 — Generar la URL pública

Ve a **Settings → Networking → Generate Domain**.  
Railway te dará una URL del tipo `https://tu-app.up.railway.app`.  
¡Comparte esa URL con quien quieras!

---

## Opción C — Publicar online con PythonAnywhere (más sencillo, sin GitHub)

PythonAnywhere tiene un plan gratuito y solo necesitas subir el ZIP.

1. Crea una cuenta gratuita en https://www.pythonanywhere.com
2. Ve a la pestaña **Files** y sube el ZIP del proyecto.
3. Abre una consola **Bash** y ejecuta:

```bash
cd ~
unzip Visual_Palladium.zip
cd "Visual Palladium"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Ve a la pestaña **Web → Add a new web app**.
5. Elige **Flask** y apunta el fichero fuente a `/home/TU_USUARIO/Visual Palladium/app.py`.
6. En **Virtualenv** pon la ruta: `/home/TU_USUARIO/Visual Palladium/venv`
7. Haz clic en **Reload** — tu app estará en `https://TU_USUARIO.pythonanywhere.com`.

---

## Estructura del proyecto

```
Visual Palladium/
├── app.py                  ← Aplicación Flask principal
├── config.py               ← Configuración (rutas, usuarios, parámetros)
├── preprocesamiento.py     ← Clases del pipeline de ML
├── requirements.txt        ← Dependencias Python
├── modelos/
│   ├── pipeline_pequeno.joblib   ← Modelo hoteles pequeños (RF)
│   ├── pipeline_mediano.joblib   ← Modelo hoteles medianos (XGBoost)
│   └── pipeline_grande.joblib   ← Modelo hoteles grandes (XGBoost)
├── datos/
│   └── reservas_simuladas.json  ← Historial de predicciones guardadas
├── entrenamiento/
│   ├── 00_Limpieza-2.ipynb      ← Notebook de limpieza y entrenamiento
│   ├── df_mediano.csv           ← Dataset hoteles medianos
│   └── df_pequeno.csv           ← Dataset hoteles pequeños
├── templates/               ← Plantillas HTML de la web
└── documentacion/           ← Documentación del proyecto
```

> El CSV original de reservas (`Reservas_22_23.csv`) y el dataset grande  
> (`df_grande.csv`) no se incluyen en el ZIP por su tamaño (>400 MB).  
> La app funciona perfectamente sin ellos.
