FROM jupyter/base-notebook:584f43f06586

# Install dependencies from Conda
RUN conda install -c conda-forge \
      cmake \
      gxx_linux-64 \
      make && \
    conda clean -a -q -y

# Copy the repository into the container
COPY --chown=${NB_UID} . /opt/geolib4d

# Build and install the project
RUN conda run -n base python -m pip install /opt/geolib4d

# Make JupyterLab the default for this application
ENV JUPYTER_ENABLE_LAB=yes
