FROM ghcr.io/fenics/dolfinx/lab:stable

WORKDIR /tmp/

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
ENV PYVISTA_OFF_SCREEN="false"
ENV PYVISTA_JUPYTER_BACKEND="static"
ENV DISPLAY=":99.0"
# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-dev mesa-utils

ADD pyproject.toml /tmp/pyproject.toml
RUN python3 -m pip install --no-cache-dir .
RUN python3 -m pip cache purge

# RUN jupyter lab build
# ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
ENTRYPOINT [ "/bin/bash" ]