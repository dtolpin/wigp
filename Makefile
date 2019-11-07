all: selfcheck

selfcheck: wigp
	./wigp selfcheck

wigp: kernel/ad/kernel.go model/ad/model.go main.go
	go build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./wigp {kernel,model}/ad/*.go 
