# An equals sign produced after the source word that supplied the colon remains
# part of a prerequisite name. A fully generated target-specific assignment is
# retained beside it to pin the other side of the partial-expansion boundary.
COLON = :
EQ = =
ASSIGNMENT = assigned: VALUE=kept

test$(COLON) one$(EQ)two
	@echo test

one$(EQ)two:
	@echo prerequisite

$(ASSIGNMENT)
assigned:
	@echo $(VALUE)
