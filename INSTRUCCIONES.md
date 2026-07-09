# Palladium — Predictor de Cancelaciones

Aplicación web para predecir la probabilidad de cancelación de reservas hoteleras.  
Se abre en el navegador; solo necesitas Python instalado.

---

## Opción A — Descargar y ejecutar en tu propio ordenador (con Git)

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

### 2. Instalar Git (si no lo tienes)

- **Windows**: descarga desde https://git-scm.com/download/win e instala con las opciones por defecto (dale a "Siguiente" en todo).
- **Mac**: abre la app **Terminal** y escribe `git --version`. Si no lo tienes, macOS te ofrecerá instalarlo automáticamente. (O descárgalo de https://git-scm.com/download/mac)

Para comprobar que está instalado:
```
git --version
```

---

### 3. Aceptar la invitación al repositorio

El proyecto es **privado**, así que primero tienes que aceptar la invitación:

1. Necesitas una cuenta en https://github.com (gratis).
2. Pídele a Erik que te invite con tu **usuario de GitHub**.
3. Acepta la invitación entrando aquí: **https://github.com/eriikvidal/tfm-palladium/invitations**  
   (o desde el email que te llega, o el banner amarillo al entrar al repo).

---

### 4. Descargar el proyecto (clonar)

Abre una terminal, ve a donde quieras guardarlo (por ejemplo el Escritorio) y clónalo:

**Mac / Linux:**
```bash
cd ~/Desktop
git clone https://github.com/eriikvidal/tfm-palladium.git
cd tfm-palladium
```

**Windows:**
```bash
cd %USERPROFILE%\Desktop
git clone https://github.com/eriikvidal/tfm-palladium.git
cd tfm-palladium
```

> La primera vez te pedirá **iniciar sesión en GitHub** (se abre el navegador o te pide
> usuario y contraseña). Inicia sesión con tu cuenta y ya tendrás acceso.  
> Si te pide "contraseña" en la terminal y no funciona, es porque GitHub ya no acepta
> contraseñas ahí: usa el botón de iniciar sesión con el navegador que aparece.

---

### 5. Crear el entorno virtual e instalar dependencias

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

### 6. Arrancar la aplicación

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
