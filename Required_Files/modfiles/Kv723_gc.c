/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__Kv723
#define _nrn_initial _nrn_initial__Kv723
#define nrn_cur _nrn_cur__Kv723
#define _nrn_current _nrn_current__Kv723
#define nrn_jacob _nrn_jacob__Kv723
#define nrn_state _nrn_state__Kv723
#define _net_receive _net_receive__Kv723 
#define _f_rates _f_rates__Kv723 
#define rates rates__Kv723 
#define states states__Kv723 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gkbar _p[0]
#define gkbar_columnindex 0
#define i _p[1]
#define i_columnindex 1
#define ik _p[2]
#define ik_columnindex 2
#define gk _p[3]
#define gk_columnindex 3
#define minf _p[4]
#define minf_columnindex 4
#define tau1 _p[5]
#define tau1_columnindex 5
#define tau2 _p[6]
#define tau2_columnindex 6
#define tadjtau _p[7]
#define tadjtau_columnindex 7
#define m1 _p[8]
#define m1_columnindex 8
#define m2 _p[9]
#define m2_columnindex 9
#define ek _p[10]
#define ek_columnindex 10
#define Vhalf1 _p[11]
#define Vhalf1_columnindex 11
#define Dtau1 _p[12]
#define Dtau1_columnindex 12
#define z1 _p[13]
#define z1_columnindex 13
#define tau01 _p[14]
#define tau01_columnindex 14
#define Vhalf2 _p[15]
#define Vhalf2_columnindex 15
#define Dtau2 _p[16]
#define Dtau2_columnindex 16
#define z2 _p[17]
#define z2_columnindex 17
#define tau02 _p[18]
#define tau02_columnindex 18
#define alpha1 _p[19]
#define alpha1_columnindex 19
#define beta1 _p[20]
#define beta1_columnindex 20
#define alpha2 _p[21]
#define alpha2_columnindex 21
#define beta2 _p[22]
#define beta2_columnindex 22
#define v0 _p[23]
#define v0_columnindex 23
#define frt _p[24]
#define frt_columnindex 24
#define Dm1 _p[25]
#define Dm1_columnindex 25
#define Dm2 _p[26]
#define Dm2_columnindex 26
#define v _p[27]
#define v_columnindex 27
#define _g _p[28]
#define _g_columnindex 28
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_gsat(void);
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_Kv723", _hoc_setdata,
 "gsat_Kv723", _hoc_gsat,
 "rates_Kv723", _hoc_rates,
 0, 0
};
#define gsat gsat_Kv723
 extern double gsat( _threadargsprotocomma_ double );
 
static void _check_rates(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_rates(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define Dtaumult2 Dtaumult2_Kv723
 double Dtaumult2 = 6;
#define Dtaumult1 Dtaumult1_Kv723
 double Dtaumult1 = 6;
#define FoverR FoverR_Kv723
 double FoverR = 11.6045;
#define Vshift Vshift_Kv723
 double Vshift = 0;
#define Vhalf Vhalf_Kv723
 double Vhalf = -50;
#define gamma gamma_Kv723
 double gamma = 0.5;
#define kV kV_Kv723
 double kV = 40;
#define k k_Kv723
 double k = 9;
#define q10tau q10tau_Kv723
 double q10tau = 5;
#define temp0 temp0_Kv723
 double temp0 = 273;
#define temptau temptau_Kv723
 double temptau = 22;
#define taudiv taudiv_Kv723
 double taudiv = 1;
#define tau0mult tau0mult_Kv723
 double tau0mult = 0.2;
#define usetable usetable_Kv723
 double usetable = 1;
#define vmax vmax_Kv723
 double vmax = 100;
#define vmin vmin_Kv723
 double vmin = -100;
#define v0erev v0erev_Kv723
 double v0erev = 65;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_Kv723", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "k_Kv723", "mV",
 "Vhalf_Kv723", "mV",
 "Vshift_Kv723", "mV",
 "v0erev_Kv723", "mV",
 "kV_Kv723", "mV",
 "temptau_Kv723", "degC",
 "vmin_Kv723", "mV",
 "vmax_Kv723", "mV",
 "temp0_Kv723", "degC",
 "FoverR_Kv723", "degC/mV",
 "gkbar_Kv723", "S/cm2",
 "i_Kv723", "mA/cm2",
 "ik_Kv723", "mA/cm2",
 "gk_Kv723", "S/cm2",
 "tau1_Kv723", "ms",
 "tau2_Kv723", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double m20 = 0;
 static double m10 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "k_Kv723", &k_Kv723,
 "Vhalf_Kv723", &Vhalf_Kv723,
 "Vshift_Kv723", &Vshift_Kv723,
 "v0erev_Kv723", &v0erev_Kv723,
 "kV_Kv723", &kV_Kv723,
 "gamma_Kv723", &gamma_Kv723,
 "temptau_Kv723", &temptau_Kv723,
 "q10tau_Kv723", &q10tau_Kv723,
 "taudiv_Kv723", &taudiv_Kv723,
 "Dtaumult1_Kv723", &Dtaumult1_Kv723,
 "Dtaumult2_Kv723", &Dtaumult2_Kv723,
 "tau0mult_Kv723", &tau0mult_Kv723,
 "vmin_Kv723", &vmin_Kv723,
 "vmax_Kv723", &vmax_Kv723,
 "temp0_Kv723", &temp0_Kv723,
 "FoverR_Kv723", &FoverR_Kv723,
 "usetable_Kv723", &usetable_Kv723,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Kv723",
 "gkbar_Kv723",
 0,
 "i_Kv723",
 "ik_Kv723",
 "gk_Kv723",
 "minf_Kv723",
 "tau1_Kv723",
 "tau2_Kv723",
 "tadjtau_Kv723",
 0,
 "m1_Kv723",
 "m2_Kv723",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 29, _prop);
 	/*initialize range parameters*/
 	gkbar = 0.0067;
 	_prop->param = _p;
 	_prop->param_size = 29;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Kv723_gc_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 29, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kv723 Kv723_gc.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_minf;
 static double *_t_tau1;
 static double *_t_tau2;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rates(_threadargsprotocomma_ double);
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_rates(_threadargsprotocomma_ double _lv);
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm1 = ( minf - m1 ) / tau1 ;
   Dm2 = ( minf - m2 ) / tau2 ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm1 = Dm1  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau1 )) ;
 Dm2 = Dm2  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau2 )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m1 = m1 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau1)))*(- ( ( ( minf ) ) / tau1 ) / ( ( ( ( - 1.0 ) ) ) / tau1 ) - m1) ;
    m2 = m2 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau2)))*(- ( ( ( minf ) ) / tau2 ) / ( ( ( ( - 1.0 ) ) ) / tau2 ) - m2) ;
   }
  return 0;
}
 static double _mfac_rates, _tmin_rates;
  static void _check_rates(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  static double _sav_gamma;
  static double _sav_k;
  static double _sav_Vhalf;
  static double _sav_Vshift;
  static double _sav_taudiv;
  static double _sav_Dtaumult1;
  static double _sav_Dtaumult2;
  static double _sav_tau0mult;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_sav_gamma != gamma) { _maktable = 1;}
  if (_sav_k != k) { _maktable = 1;}
  if (_sav_Vhalf != Vhalf) { _maktable = 1;}
  if (_sav_Vshift != Vshift) { _maktable = 1;}
  if (_sav_taudiv != taudiv) { _maktable = 1;}
  if (_sav_Dtaumult1 != Dtaumult1) { _maktable = 1;}
  if (_sav_Dtaumult2 != Dtaumult2) { _maktable = 1;}
  if (_sav_tau0mult != tau0mult) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rates =  vmin ;
   _tmax =  vmax ;
   _dx = (_tmax - _tmin_rates)/199.; _mfac_rates = 1./_dx;
   for (_i=0, _x=_tmin_rates; _i < 200; _x += _dx, _i++) {
    _f_rates(_p, _ppvar, _thread, _nt, _x);
    _t_minf[_i] = minf;
    _t_tau1[_i] = tau1;
    _t_tau2[_i] = tau2;
   }
   _sav_celsius = celsius;
   _sav_gamma = gamma;
   _sav_k = k;
   _sav_Vhalf = Vhalf;
   _sav_Vshift = Vshift;
   _sav_taudiv = taudiv;
   _sav_Dtaumult1 = Dtaumult1;
   _sav_Dtaumult2 = Dtaumult2;
   _sav_tau0mult = tau0mult;
  }
 }

 static int rates(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_rates(_p, _ppvar, _thread, _nt);
#endif
 _n_rates(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_rates(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rates(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_rates * (_lv - _tmin_rates);
 if (isnan(_xi)) {
  minf = _xi;
  tau1 = _xi;
  tau2 = _xi;
  return;
 }
 if (_xi <= 0.) {
 minf = _t_minf[0];
 tau1 = _t_tau1[0];
 tau2 = _t_tau2[0];
 return; }
 if (_xi >= 199.) {
 minf = _t_minf[199];
 tau1 = _t_tau1[199];
 tau2 = _t_tau2[199];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 minf = _t_minf[_i] + _theta*(_t_minf[_i+1] - _t_minf[_i]);
 tau1 = _t_tau1[_i] + _theta*(_t_tau1[_i+1] - _t_tau1[_i]);
 tau2 = _t_tau2[_i] + _theta*(_t_tau2[_i+1] - _t_tau2[_i]);
 }

 
static int  _f_rates ( _threadargsprotocomma_ double _lv ) {
   if ( gamma  == 0.5 ) {
     z1 = 2.8 ;
     Vhalf1 = - 49.8 + Vshift ;
     tau01 = 20.7 * tau0mult ;
     Dtau1 = 176.1 * Dtaumult1 ;
     z2 = 8.9 ;
     Vhalf2 = - 55.5 + Vshift ;
     tau02 = 149.0 * tau0mult ;
     Dtau2 = 1473.0 * Dtaumult2 ;
     }
   if ( gamma  == 1.0 ) {
     z1 = 3.6 ;
     Vhalf1 = - 25.3 + Vshift ;
     tau01 = 29.2 * tau0mult ;
     Dtau1 = 74.6 * Dtaumult1 ;
     z2 = 9.8 ;
     Vhalf2 = - 44.7 + Vshift ;
     tau02 = 155.0 * tau0mult ;
     Dtau2 = 549.0 * Dtaumult2 ;
     }
   tadjtau = pow( q10tau , ( ( celsius - temptau ) / 10.0 ) ) ;
   frt = FoverR / ( temp0 + celsius ) ;
   alpha1 = exp ( z1 * gamma * frt * ( _lv - Vhalf1 ) ) ;
   beta1 = exp ( - z1 * ( 1.0 - gamma ) * frt * ( _lv - Vhalf1 ) ) ;
   tau1 = ( Dtau1 / ( alpha1 + beta1 ) + tau01 ) / ( tadjtau * taudiv ) ;
   alpha2 = exp ( z2 * gamma * frt * ( _lv - Vhalf2 ) ) ;
   beta2 = exp ( - z2 * ( 1.0 - gamma ) * frt * ( _lv - Vhalf2 ) ) ;
   tau2 = ( Dtau2 / ( alpha2 + beta2 ) + tau02 ) / ( tadjtau * taudiv ) ;
   minf = 1.0 / ( 1.0 + exp ( - ( _lv - Vhalf - Vshift ) / k ) ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double gsat ( _threadargsprotocomma_ double _lv ) {
   double _lgsat;
 _lgsat = 1.0 ;
   v0 = v0erev + ek ;
   if ( _lv > v0 ) {
     _lgsat = 1.0 + ( v0 - _lv + kV * ( 1.0 - exp ( - ( _lv - v0 ) / kV ) ) ) / ( _lv - ek ) ;
     }
   
return _lgsat;
 }
 
static void _hoc_gsat(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  gsat ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  m2 = m20;
  m1 = m10;
 {
   rates ( _threadargscomma_ v ) ;
   m1 = minf ;
   m2 = minf ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];

#if 0
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gk = gkbar * gsat ( _threadargscomma_ v ) * ( pow( m1 , 2.0 ) ) * m2 ;
   ik = gk * ( v - ek ) ;
   i = ik ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ek = _ion_ek;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ek = _ion_ek;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m1_columnindex;  _dlist1[0] = Dm1_columnindex;
 _slist1[1] = m2_columnindex;  _dlist1[1] = Dm2_columnindex;
   _t_minf = makevector(200*sizeof(double));
   _t_tau1 = makevector(200*sizeof(double));
   _t_tau2 = makevector(200*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "Kv723_gc.mod";
static const char* nmodl_file_text = 
  ": M conductance\n"
  ": from Mateos-Aparicio et al (2014)\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX Kv723\n"
  "	USEION k READ ek WRITE ik\n"
  "	RANGE gkbar, ik, minf, tau1, tau2, i, gk, m1, m2, tadjtau\n"
  "	GLOBAL Vhalf, Vshift, k, v0erev, kV, gamma\n"
  "	GLOBAL Dtaumult1, Dtaumult2, tau0mult, taudiv, q10tau\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(S) = (siemens)\n"
  "	(um) = (micron)\n"
  "} \n"
  "\n"
  "PARAMETER {\n"
  "	gkbar = 0.0067 			(S/cm2)\n"
  "	k = 9           		(mV)\n"
  "	Vhalf = -50             (mV)  :for minf(V)\n"
  "	Vshift = 0              (mV)	:for g(V) and minf(V)     \n"
  "	v0erev = 65             (mV)     :50-80\n"
  "	kV = 40                 (mV)     \n"
  "	gamma = 0.5                      :0.5,1\n"
  "\n"
  "	temptau = 22	          (degC) :tau reference temperature 	\n"
  "	q10tau  = 5\n"
  "	taudiv = 1\n"
  "	Dtaumult1 = 6  \n"
  "	Dtaumult2 = 6 \n"
  "	tau0mult = 0.2\n"
  "\n"
  "	vmin = -100	            (mV)\n"
  "	vmax = 100	            (mV)\n"
  "	temp0 = 273		          (degC)\n"
  "	FoverR = 11.6045039552	(degC/mV)\n"
  "} \n"
  " \n"
  "ASSIGNED {\n"
  "	ek		(mV)\n"
  "	v 	     	(mV)\n"
  "	celsius		(degC)\n"
  "	Vhalf1    (mV) \n"
  "	Dtau1     (ms)\n"
  "	z1               \n"
  "	tau01   	(ms)	 \n"
  "	Vhalf2  	(mV)	  \n"
  "	Dtau2   	(ms)  \n"
  "	z2               \n"
  "	tau02   	(ms)	  \n"
  "	alpha1				  \n"
  "	beta1	  		  \n"
  "	alpha2		\n"
  "	beta2	\n"
  "	i 	    	(mA/cm2)\n"
  "	ik 	     	(mA/cm2)\n"
  "	gk		      (S/cm2)\n"
  "	minf\n"
  "	v0        (mV)      \n"
  "	tau1			(ms)\n"
  "	tau2			(ms)\n"
  "	tadjtau\n"
  "	frt		    (/mV)\n"
  "}\n"
  " \n"
  "STATE { m1 m2 }\n"
  "\n"
  "INITIAL { \n"
  "	rates(v)\n"
  "	m1 = minf\n"
  "	m2 = minf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "  SOLVE states METHOD cnexp\n"
  "	gk = gkbar*gsat(v)*(m1^2)*m2\n"
  "	ik = gk*(v - ek)\n"
  "	i = ik\n"
  "} \n"
  "\n"
  "DERIVATIVE states {\n"
  "	rates(v)\n"
  "	m1' = (minf - m1)/tau1\n"
  "	m2' = (minf - m2)/tau2\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) {\n"
  "  TABLE minf, tau1, tau2\n"
  "	DEPEND celsius, gamma, k, Vhalf, Vshift, taudiv, Dtaumult1, Dtaumult2, tau0mult\n"
  "	FROM vmin TO vmax WITH 199\n"
  "	\n"
  "  IF (gamma == 0.5) {\n"
  "  	z1 = 2.8\n"
  "		Vhalf1 = -49.8+Vshift 	:(mV)  shifted - 20 mV (when Vshift = 0)\n"
  "		tau01 = 20.7 (ms) *tau0mult	  :(ms)\n"
  "		Dtau1 = 176.1 (ms) *Dtaumult1	:(ms)\n"
  "		z2 = 8.9	              \n"
  "		Vhalf2 = -55.5+Vshift 					:(mV)  shifted - 20 mV\n"
  "		tau02 = 149 (ms) *tau0mult   					:(ms)\n"
  "		Dtau2 = 1473 (ms) *Dtaumult2 	  			:(ms)\n"
  "	}	\n"
  "	IF (gamma == 1) {\n"
  "  	z1 = 3.6\n"
  "		Vhalf1 = -25.3+Vshift		:(mV)  shifted - 20 mV\n"
  "		tau01 = 29.2 (ms) *tau0mult	  :(ms)\n"
  "		Dtau1 = 74.6 (ms) *Dtaumult1	:(ms)\n"
  "		z2 = 9.8	\n"
  "		Vhalf2 = -44.7+Vshift 					:(mV)  shifted - 20 mV\n"
  "		tau02 = 155 (ms) *tau0mult   					:(ms)\n"
  "		Dtau2 = 549 (ms) *Dtaumult2  	  			:(ms)\n"
  "	}\n"
  "  tadjtau = q10tau^((celsius - temptau)/10 (degC))\n"
  "	frt = FoverR/(temp0 + celsius)\n"
  "\n"
  "  alpha1 = exp(z1*gamma*frt*(v - Vhalf1))\n"
  "  beta1 = exp(-z1*(1-gamma)*frt*(v - Vhalf1))\n"
  "  tau1 = (Dtau1/(alpha1 + beta1) + tau01)/(tadjtau*taudiv)\n"
  "  \n"
  "  alpha2 = exp(z2*gamma*frt*(v - Vhalf2))\n"
  "  beta2 = exp(-z2*(1-gamma)*frt*(v - Vhalf2))\n"
  "  tau2 = (Dtau2/(alpha2 + beta2) + tau02)/(tadjtau*taudiv)\n"
  "\n"
  "  minf = 1/(1 + exp(-(v - Vhalf - Vshift)/k))\n"
  "}\n"
  "\n"
  "FUNCTION gsat (v (mV)) {\n"
  "	gsat = 1\n"
  "	v0 = v0erev + ek  \n"
  "	IF (v > v0) {\n"
  "		gsat = 1+(v0-v+kV*(1-exp(-(v-v0)/kV)))/(v-ek)\n"
  "	}\n"
  "}\n"
  "\n"
  "\n"
  ;
#endif
