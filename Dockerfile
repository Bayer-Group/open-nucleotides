FROM mambaorg/micromamba:0.23.3

WORKDIR /tmp

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-production.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes

ENV NUCLEOTIDES_ENVIRONMENT=production

ARG version
ARG builddate
ARG github_sha

COPY --chown=$MAMBA_USER:$MAMBA_USER setup.py .
COPY --chown=$MAMBA_USER:$MAMBA_USER nucleotides nucleotides
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$github_sha
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "nucleotides.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.label-schema.schema-version="1.0" \
    org.label-schema.build-date=$builddate \
    org.label-schema.version=$version
