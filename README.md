# Multi-document summarization

Library and command line utility for generating summary from docment set. The
package also contains simple evaluation framework for text summaries.

## Requirement

Make sure you have [python](http://www.python.org/) 2.7+ and 
[jieba](https://github.com/fxsjy/jieba), 
[nltk](http://www.nltk.org/),
[pulp](https://pythonhosted.org/PuLP/),
[GLPK](https://www.gnu.org/software/glpk/),
[docopt](http://docopt.org/)

## Usage

```sh
$ python -m nlp_sum submodular --length 200 --language english --stem --output ~/Desktop/out --file ~/Desktop/all
$ python -m nlp_sum lexrank --query summarization --length 200 --language english --stem --output ~/Desktop/out --file ~/Desktop/all
$ python -m nlp_sum --help # for more info
```

```docker
$ docker pull zjn1993/nlp_sum # pull image
$ cd nlp_sum/docker && docker build .
```
