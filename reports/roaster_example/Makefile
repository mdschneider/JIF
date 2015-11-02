################################################################################
# Copyright 2011
# Andrew Redd
# 11/23/2011
#
# Description of File:
# Makefile for knitr compiling
#
################################################################################
all:pdf  # default rule DO NOT EDIT
################################################################################
MAINFILE  := roaster_example
RNWFILES  :=
RFILES    :=
TEXFILES  :=
CACHEDIR  := cache
FIGUREDIR := figures
LATEXMK_FLAGS :=
##### Explicit Dependencies #####
################################################################################
RNWTEX = $(RNWFILES:.Rnw=.tex)
ROUTFILES = $(RFILES:.R=.Rout)
RDATAFILES= $(RFILES:.R=.Rdata)
MAINTEX = $(MAINFILE:=.tex)
MAINPDF = $(MAINFILE:=.pdf)
ALLTEX = $(MAINTEX) $(RNWTEX) $(TEXFILES)

# Dependencies
$(RNWTEX): $(RDATAFILES)
$(MAINTEX): $(RNWTEX) $(TEXFILES)
$(MAINPDF): $(MAINTEX) $(ALLTEX)

.PHONY:pdf tex clean clearcache cleanall
pdf: $(MAINPDF)
tex: $(RDATAFILES) $(ALLTEX)

$(CACHEDIR):
	mkdir $(CACHEDIR)

$(FIGUREDIR):
	mkdir $(FIGUREDIR)

%.tex:%.Rnw
	Rscript \
	  -e "library(knitr)" \
	  -e "knitr::opts_chunk[['set']](fig.path='$(FIGUREDIR)/$*-')" \
	  -e "knitr::opts_chunk[['set']](cache.path='$(CACHEDIR)/$*-')" \
	  -e "knitr::knit('$<','$@')"


%.R:%.Rnw
	Rscript -e "Sweave('$^', driver=Rtangle())"

%.Rout:%.R
	R CMD BATCH "$^" "$@"

%.pdf: %.tex
	@ ./add_git_tag.sh
	@ latexmk -pdf $<
	@ open -a Skim $(MAINPDF)

clean:
	-latexmk -c -quiet $(MAINFILE).tex
	-rm -f $(MAINTEX) $(RNWTEX)
	-rm -rf $(FIGUREDIR)
	-rm *tikzDictionary
	-rm $(MAINPDF)
	-rm -f git_tag.tex

clearcache:
	-rm -rf cache

cleanall: clean clearcache
