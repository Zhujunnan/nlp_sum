FROM ubuntu:wily

MAINTAINER junnanzhu

COPY sources.list /etc/apt/

RUN apt-get update --fix-missing && \
    apt-get install -y \
        build-essential \
        bzip2 \
        unzip \
        cmake \
        git \
        wget \
        curl \
        grep \
        sed \
        vim \
        gfortran \
        pkg-config \
        ca-certificates

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV ANACONDA_ROOT /opt/anaconda
ENV PATH $ANACONDA_ROOT/bin:$PATH

RUN wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh -P /tmp/
RUN /bin/bash /tmp/Anaconda2-4.3.1-Linux-x86_64.sh -b -p $ANACONDA_ROOT

RUN pip install -U nltk
RUN python -m nltk.downloader -q punkt
RUN pip install jieba
RUN pip install pulp
RUN pip install docopt

WORKDIR /usr/local

RUN wget http://ftp.gnu.org/gnu/glpk/glpk-4.57.tar.gz \
	&& tar -zxvf glpk-4.57.tar.gz

WORKDIR /usr/local/glpk-4.57

RUN ./configure \
	&& make \
	&& make check \
	&& make install \
	&& make distclean \
	&& ldconfig \
# Cleanup
	&& rm -rf /user/local/glpk-4.57.tar.gz \
	&& apt-get clean

RUN rm /tmp/Anaconda2-4.3.1-Linux-x86_64.sh

WORKDIR /root

RUN git clone https://github.com/Zhujunnan/nlp_sum.git

CMD [ "/bin/bash" ]
