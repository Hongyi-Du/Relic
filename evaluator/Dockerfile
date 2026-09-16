# Source-derived local evaluator environment for the public Relic OSS packs.
#
# Ported from SocioGenesis hci-human-seat@dda36fb563375060ae8d8850300db01eb4695d29
# `.evaluator_image/Dockerfile`.  This image contains only the Python test
# runtime dependencies.  It does not copy a benchmark, hidden suite, model
# credential, relay configuration, controller source, or an author-published
# evaluator binding into the image.
#
# A locally built image is useful for source-closure and qualification checks,
# but it is not the immutable, author-published evaluator image required for a
# paper reproduction.  `relic run-main` deliberately rejects its local image
# ID as a formal binding.
FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# The suite runner and the libraries the upstream AnyIO-era image carried.
RUN pip install --no-cache-dir \
      pytest==8.3.4 pytest-asyncio==0.25.0 pytest-mock trio==0.27.0 \
      typing_extensions attrs sortedcontainers outcome sniffio idna

# Test-runner plugins declared by public tests.  The evaluator runs tests with
# project addopts cleared where required, but the readiness probe still checks
# that declared dependencies can be imported in the sandbox.
RUN pip install --no-cache-dir \
      pytest-benchmark coverage pytest-xdist

# Per pack, install only libraries imported by the frozen hidden suites.
RUN pip install --no-cache-dir beautifulsoup4
RUN pip install --no-cache-dir \
      mypy_extensions pathspec platformdirs pytokens click
RUN pip install --no-cache-dir \
      annotated-types typing-inspection pydantic-core==2.47.0
RUN pip install --no-cache-dir \
      multidict yarl frozenlist propcache aiohappyeyeballs aiosignal
RUN pip install --no-cache-dir hypothesis immutables
# celery declares only click and kombu, but importing celery.app pulls this
# chain.  The product under test is not installed; only its imports are.
RUN pip install --no-cache-dir \
      kombu python-dateutil billiard vine tzdata tzlocal \
      click-didyoumean click-plugins click-repl

# Candidate packs from the frozen Relic main benchmark.
RUN pip install --no-cache-dir \
      openpyxl python-docx reportlab pypdf pdfplumber
RUN pip install --no-cache-dir \
      fastapi pydantic-settings email-validator sqlalchemy httpx \
      "python-jose[cryptography]" "passlib[bcrypt]" python-multipart
# fastapi_dashboard declares uvicorn and psycopg2-binary.  The readiness probe
# checks declared dependencies before product code is run.
RUN pip install --no-cache-dir uvicorn psycopg2-binary
RUN pip install --no-cache-dir flask numpy opencv-python-headless
# passlib 1.7.4 reads bcrypt.__about__, removed by bcrypt 4.1.  The frozen
# FastAPI pack pins 4.0.1 for that reason.
RUN pip install --no-cache-dir "bcrypt==4.0.1"
