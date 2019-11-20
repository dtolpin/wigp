all: wigp gen nlpd

selfcheck: wigp
	./wigp selfcheck

wigp: kernel/ad/kernel.go priors/ad/priors.go model/model.go main.go 
	go build .

gen: cmd/gen/main.go
	go build ./cmd/gen

nlpd: cmd/nlpd/main.go
	go build ./cmd/nlpd

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

priors/ad/priors.go: priors/priors.go
	deriv priors

clean:
	rm -f ./wigp {kernel,priors}/ad/*.go 
