# DOLFINx documentation
A thorough introduction to DOLFINx.

This documentation is experimental and likely to change.

```{tableofcontents}
```


## Build dependencies
To build a docker image for this documentation, you can run
```bash
docker build -t fenicsx-docs -f docker/Dockerfile docker/
```
from the root of this repository.
To create a one-time usage container you can call:
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm test_docs
```

## Citing the documentation

