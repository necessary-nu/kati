# A fully expanded rule can carry a target-specific assignment, while an equals
# sign expanded in a later prerequisite word stays in that prerequisite's name.
TSV:=test: A=PASS
A_EQ_B:=A=B
EQ==
$(TSV)
test: A$(EQ)B

$(A_EQ_B):
	echo $(A)
