JL          = ~/julia/julia
OPTS        = "-t 8"
OBJS        = main.jl

main: $(OBJS)
	$(JL) $(OPTS) $(OBJS)

clean:
	-rm -f *.txt *.png *.dat nohup.out
	-rm -rf data
