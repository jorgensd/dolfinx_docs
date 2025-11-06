FROM ghcr.io/fenics/dolfinx/lab:nightly

RUN apt-get update && apt-get install -y libgl1-mesa-dev mesa-utils
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
ENV PYVISTA_OFF_SCREEN="false"
ENV PYVISTA_JUPYTER_BACKEND="html"


ADD pyproject.toml ./pyproject.toml
RUN python3 -m pip install .

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
# 24.04 adds `ubuntu` as uid 1000;
# remove it if it already exists before creating our user
RUN id -nu ${NB_UID} && userdel --force $(id -nu ${NB_UID}) || true; \
    useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME=/home/${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []