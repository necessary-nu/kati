$(shell exit 5)

NAMEWORKAROUND := .SHELLSTATUS
testTargetWithShellCommand:
	@echo $(shell exit 7)
	@echo $($(NAMEWORKAROUND))

test: testTargetWithShellCommand
	@# Suppress the "Nothing to be done for "test"." message
	@:
