
/*    Generating New Variables    */

clear all
set more off

*cd C:\Users\pb061\Documents\Nutstore\EconometricsTermPaper\step1_CutDataSet
cd C:\Users\lenovo\Documents\我的坚果云\EconometricsTermPaper\step1_CutDataSet
use temp,clear
xtset symbol year
gen WC_acc = ((d.currentassets-d.monetaryfunds)-(d.currentliabilities)-d.suodesui)/(0.5*(l.assets+assets))
gen ch_rev = d.recivables/(0.5*(l.assets+assets))
gen ch_inv = d.inventory/(0.5*(l.assets+assets))
gen soft_asset = (assets - ppe)/ assets

rename assetsgr ch_assets
rename ppegr ch_ppe

* Generating DA by jones model *
gen js1 = 1/l.assets
gen js2 = (d.sales-d.recivables)/l.assets
gen js3 = d.ppe/l.assets
gen acc = accurals/assets
bys nnindcd: reg acc js*
predict da,r
drop js*
sort symbol year
****
* performance vars
gen ch_cs = sales - d.recivables
gen ch_roa =  netprofit/(0.5*(l.assets+assets)) - l.netprofit/(0.5*(l.assets+assets))
rename salesgr ch_sales

* leverages
gen lev = debts/assets

* non-financial vars
gen ch_emp = l.nemployees / nemployees 
gen llemon = l.lemon 
gen preout = 0 
replace preout = 1 if (l.netprofit<0&l.l.netprofit<0) 
lab var preout "连续两年净利润为负数有退市可能"

drop if year<2002

keep symbol year lemon pe marketvalue tobinq booktomarket nnindnme nnindcd equitynatureid ch* accurals parttime WC_acc preout soft_asset lev llemon da netprofit


gen ind2 = .
replace ind2 = 1 if nnindnme == "农、林、牧、渔服务业"|nnindnme == "农业"|nnindnme == "林业"|nnindnme == "渔业" |nnindnme == "畜牧业"

replace ind2 = 2 if nnindnme == "农副食品加工业" | nnindnme == "化学原料及化学制品制造业"| nnindnme == "化学纤维制造业"
replace ind2 = 2 if nnindnme == "医药制造业" | nnindnme == "土木工程建筑业" | nnindnme == "有色金属冶炼及压延加工业" | nnindnme == "有色金属矿采选业" 
replace ind2 = 2 if nnindnme == "木材加工及木、竹、藤、棕、草制品业" |nnindnme == "橡胶和塑料制品业" |nnindnme == "汽车制造业" 
replace ind2 = 2 if nnindnme == "煤炭开采和洗选业" |nnindnme == "电力、热力生产和供应业" |nnindnme == "电气机械及器材制造业" 
replace ind2 = 2 if nnindnme == "皮革、毛皮、羽毛及其制品和制鞋业" |nnindnme == "石油加工、炼焦及核燃料加工业" |nnindnme == "石油和天然气开采业"
replace ind2 = 2 if nnindnme == "纺织业" |nnindnme == "非金属矿物制品业" |nnindnme == "非金属矿采选业" 
replace ind2 = 2 if nnindnme == "黑色金属矿采选业" |nnindnme == "黑色金属冶炼及压延加工业" 

replace ind2 = 4 if nnindnme == "其他金融业" | nnindnme == "保险业" |nnindnme == "货币金融服务" | nnindnme == "资本市场服务"

replace ind2 = 3 if mi(ind2)

label var ind2 "按三产业划分" 

replace equitynatureid = "0" if equitynatureid!="1"
drop WC_acc nnindnme nnindcd
mvencode _all, mv(0) override

export excel using temp2.xlsx, replace first(var)
