# Evidence and rerun inputs

This ZIP contains implementation, experiment orchestration and tests. Large
checkpoints, frozen evaluation inputs and full experiment registries are not
included. Relative `artifacts/` paths in selected runners are intended for
those separately distributed assets; they are not silently generated.

For a paper-number reproduction release, publish a separate versioned
evidence bundle with: accepted summary rows; input, checkpoint and source
hashes; protocol manifests; command lines; and a map from each paper table or
figure to its generating script. This code ZIP can be reviewed and built
without downloading that larger bundle.
