# Build only after reviewing a candidate model report. RUNTIME_IMAGE must be a digest-pinned image.
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}
COPY --chown=10001:10001 models/artifacts/fraud_model.pt /models/fraud_model.pt
