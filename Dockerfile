FROM mambaorg/micromamba:2.3.2

WORKDIR /work

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba env create -f /tmp/environment.yml -y && micromamba clean -a -y

SHELL ["bash", "-lc"]
ENV PATH=/opt/conda/envs/qat-eval/bin:$PATH
EXPOSE 8888
CMD ["bash"]


