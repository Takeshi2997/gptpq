JL          = ~/julia/julia
OPTS        = "-t 8"
BASE        = functions.jl setup.jl gpcore.jl
OBJS        = main.jl
CALC        = calculation.jl

main: $(BASE) $(CORE) $(OBJS)
	$(JL) $(OPTS) $(OBJS)

calc: $(BASE) $(CORE) $(CALC)
	$(JL) $(OPTS) $(CALC)

clean:
	-rm -f *.txt *.png *.dat nohup.out
	-rm -rf data
