README.md


# Create database for hyperparameter optimization
	a. sudo apt-get install postgresql libpq-dev postgresql-client postgresql-client-common #Install posetgresql with online installer
	b. sudo -i -u postgres #goes into postgres default user
	c. psql postgres #enter the postgres interface
	d. create role monkey_optim WITH LOGIN superuser password 'serrelab'; #create the admin
	e. alter role monkey_optim superuser; #ensure we are sudo
	f. create database monkey_optim with owner monkey_optim; #create the database
	g. \q #quit
	h. psql monkey_optim -h 127.0.0.1 -d monkey_optim


1. Fill this out
2. Create the network refinement script -- (after training for x/y, add aux_tasks such as z prediction, occlusion, etc. Set the learning rate specifically for different layers)