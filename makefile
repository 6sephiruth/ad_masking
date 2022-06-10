### make targets ###
%: %.py
	@python3 -u $^ params_.yaml