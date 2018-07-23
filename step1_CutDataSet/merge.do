clear all
set more off

cd C:\Users\pb061\Documents\Nutstore\EconometricsTermPaper\step1_CutDataSet

use base0607,clear

merge m:m symbol using basicinf.dta
drop if _merge == 2
drop _merge

merge m:m symbol using equity
replace equitynatureid = "4" if _merge == 1
drop _merge

merge m:m symbol year using growth
drop _merge

merge m:m symbol year using incomestates
drop if year<2000
drop if typrep =="B"
drop _merge
drop stktype


*replace monetaryfunds = 0 if mi(monetaryfunds)

*replace inventory = if mi(inventory)

save temp.dta,replace

use marketvalue,clear

merge m:m symbol year using temp.dta

sort symbol year
quietly by symbol year:  gen dup = cond(_N==1,0,_n)
drop if dup>1

drop if _merge == 2
drop _merge

merge m:m symbol year using bs2.dta
drop if _merge != 3
drop _merge

merge m:m symbol year using governance.dta
drop if _merge != 3
drop _merge


order symbol year lemon
drop dup
save temp,replace

* output!
export excel using temp.xlsx,replace first(var)

