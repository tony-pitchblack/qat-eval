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
CMD micromamba run -n qat-eval tmux new-session -d -s jupyter \
      "jupyter lab \
        --ip=0.0.0.0 \
        --no-browser \
        --NotebookApp.token='' \
        --port=\${JUPYTER_PORT:-8888}" \
  && micromamba run -n qat-eval tmux new-session -d -s mlflow \
      "mlflow ui \
        --host 0.0.0.0 \
        --port \${MLFLOW_PORT:-5000}" \
  && tail -f /dev/null

WORKDIR /work/qat-eval

RUN echo 'eval "$(micromamba shell hook -s bash)"' >> ~/.bashrc \
 && echo 'micromamba activate qat-eval' >> ~/.bashrc
