Releasing a New Version
=======================

```bash
make check test
emacs dataiter.py setup.py
emacs NEWS.md
sudo make install clean
make test-installed
tools/release
make push clean
sudo pip3 uninstall dataiter
sudo pip3 uninstall dataiter
sudo pip3 install dataiter
make test-installed
```
