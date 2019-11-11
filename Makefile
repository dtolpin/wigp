all: selfcheck

selfcheck: wigp
	./wigp selfcheck

wigp: kernel/ad/kernel.go priors/ad/priors.go model/model.go main.go 
	go build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

priors/ad/priors.go: priors/priors.go
	deriv priors

clean:
	rm -f ./wigp {kernel,priors}/ad/*.go 
