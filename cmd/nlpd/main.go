package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

var (
	COMMA = ","
	SKIP = 0
	NOISE = false
	JNOISE = -2
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`Computes average negative log predictive density. Invocation:
	%s  [OPTIONS]
`, os.Args[0])
		flag.PrintDefaults()
	}
	flag.StringVar(&COMMA, "comma", COMMA, "field separator")
	flag.IntVar(&SKIP, "s", SKIP, "initial records to skip")
	flag.BoolVar(&NOISE, "noise", NOISE, "add noise to predicted error")
	flag.IntVar(&JNOISE, "j", JNOISE, "index of the log(noise) field")
}

// negative log predictive density
func nlpd(y, mean, std float64) float64 {
	vari := std * std
	logv := math.Log(vari)
	d := y - mean
	return 0.5 * (math.Log(2) * math.Log(math.Pi) + d*d/vari + logv)
}

func main() {
	flag.Parse()

	rdr := csv.NewReader(os.Stdin)
	rdr.Comma = rune(COMMA[0])

	rdr.Read() // skip the header
	sum := 0.
	n := 0
	for ;; n++ {
		record, err := rdr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		if n < SKIP {
			continue
		}

		y, _ := strconv.ParseFloat(record[1], 64)
		mean, _ := strconv.ParseFloat(record[2], 64)
		std, _ := strconv.ParseFloat(record[3], 64)
		if NOISE {
			jnoise := JNOISE
			if jnoise < 0 {
				jnoise += len(record)
			}
			lognoise, _ := strconv.ParseFloat(record[jnoise], 64)
			std += 0.1*math.Exp(lognoise)
		}
		sum += nlpd(y, mean, std)
	}
	fmt.Printf("%f\n", sum/float64(n))
}
