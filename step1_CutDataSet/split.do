split accper,p("/")
keep if accper2 == "12"
ren accper1 Year
drop accper*
destring Year,replace
drop if Year<2000
drop if Year>2017

