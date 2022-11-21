```
python setup.py install

```


若git push出现超时，则输入：
```
git config --global http.sslVerigy "false"
git config --global --unset http.proxy
git config --global --unset https.proxy
```