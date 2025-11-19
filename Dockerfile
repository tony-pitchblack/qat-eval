FROM mambaorg/micromamba:2.3.2

USER root
RUN mkdir -p /work && chown -R $MAMBA_USER:$MAMBA_USER /work
USER $MAMBA_USER

WORKDIR /work

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba env create -f /tmp/environment.yml -y && micromamba clean -a -y
RUN micromamba run -n qat-eval git clone https://github.com/tony-pitchblack/qat-eval.git /work/qat-eval

SHELL ["bash", "-lc"]
ENV PATH=/opt/conda/envs/qat-eval/bin:$PATH
EXPOSE 8888 5000
ENV JUPYTER_PORT=8888 MLFLOW_PORT=5000

RUN echo 'eval "$(micromamba shell hook -s bash)"' >> ~/.bashrc \
 && echo 'micromamba activate qat-eval' >> ~/.bashrc

CMD ["micromamba","run","-n","qat-eval","bash"]

WORKDIR /work/qat-eval
