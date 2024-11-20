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
 
#define nrn_init _nrn_init__BK_gc
#define _nrn_initial _nrn_initial__BK_gc
#define nrn_cur _nrn_cur__BK_gc
#define _nrn_current _nrn_current__BK_gc
#define nrn_jacob _nrn_jacob__BK_gc
#define nrn_state _nrn_state__BK_gc
#define _net_receive _net_receive__BK_gc 
#define rates rates__BK_gc 
#define state state__BK_gc 
 
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
#define diff _p[0]
#define diff_columnindex 0
#define gakbar _p[1]
#define gakbar_columnindex 1
#define gabkbar _p[2]
#define gabkbar_columnindex 2
#define ik _p[3]
#define ik_columnindex 3
#define gak _p[4]
#define gak_columnindex 4
#define gabk _p[5]
#define gabk_columnindex 5
#define ainf _p[6]
#define ainf_columnindex 6
#define atau _p[7]
#define atau_columnindex 7
#define abinf _p[8]
#define abinf_columnindex 8
#define abtau _p[9]
#define abtau_columnindex 9
#define acai _p[10]
#define acai_columnindex 10
#define ab _p[11]
#define ab_columnindex 11
#define a _p[12]
#define a_columnindex 12
#define ek _p[13]
#define ek_columnindex 13
#define cai _p[14]
#define cai_columnindex 14
#define ncai _p[15]
#define ncai_columnindex 15
#define Dab _p[16]
#define Dab_columnindex 16
#define Da _p[17]
#define Da_columnindex 17
#define v _p[18]
#define v_columnindex 18
#define _g _p[19]
#define _g_columnindex 19
#define _ion_cai	*_ppvar[0]._pval
#define _ion_ek	*_ppvar[1]._pval
#define _ion_ik	*_ppvar[2]._pval
#define _ion_dikdv	*_ppvar[3]._pval
#define _ion_ncai	*_ppvar[4]._pval
 
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
 /* declaration of user functions */
 static void _hoc_peakab(void);
 static void _hoc_peaka(void);
 static void _hoc_rates(void);
 static void _hoc_shiftab(void);
 static void _hoc_shifta(void);
 static void _hoc_taufunc(void);
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
 "setdata_BK_gc", _hoc_setdata,
 "peakab_BK_gc", _hoc_peakab,
 "peaka_BK_gc", _hoc_peaka,
 "rates_BK_gc", _hoc_rates,
 "shiftab_BK_gc", _hoc_shiftab,
 "shifta_BK_gc", _hoc_shifta,
 "taufunc_BK_gc", _hoc_taufunc,
 0, 0
};
#define peakab peakab_BK_gc
#define peaka peaka_BK_gc
#define shiftab shiftab_BK_gc
#define shifta shifta_BK_gc
#define taufunc taufunc_BK_gc
 extern double peakab( _threadargsprotocomma_ double );
 extern double peaka( _threadargsprotocomma_ double );
 extern double shiftab( _threadargsprotocomma_ double );
 extern double shifta( _threadargsprotocomma_ double );
 extern double taufunc( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define base base_BK_gc
 double base = 4;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "base_BK_gc", "mV",
 "diff_BK_gc", "1",
 "gakbar_BK_gc", "S/cm2",
 "gabkbar_BK_gc", "S/cm2",
 "ik_BK_gc", "mA/cm2",
 "gak_BK_gc", "S/cm2",
 "gabk_BK_gc", "S/cm2",
 "atau_BK_gc", "ms",
 "abtau_BK_gc", "ms",
 "acai_BK_gc", "mM",
 0,0
};
 static double a0 = 0;
 static double ab0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "base_BK_gc", &base_BK_gc,
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
 
#define _cvode_ieq _ppvar[5]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"BK_gc",
 "diff_BK_gc",
 "gakbar_BK_gc",
 "gabkbar_BK_gc",
 0,
 "ik_BK_gc",
 "gak_BK_gc",
 "gabk_BK_gc",
 "ainf_BK_gc",
 "atau_BK_gc",
 "abinf_BK_gc",
 "abtau_BK_gc",
 "acai_BK_gc",
 0,
 "ab_BK_gc",
 "a_BK_gc",
 0,
 0};
 static Symbol* _ca_sym;
 static Symbol* _k_sym;
 static Symbol* _nca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 20, _prop);
 	/*initialize range parameters*/
 	diff = 100000;
 	gakbar = 0.0624;
 	gabkbar = 0.0156;
 	_prop->param = _p;
 	_prop->param_size = 20;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 6, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[1]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 prop_ion = need_memb(_nca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[4]._pval = &prop_ion->param[1]; /* ncai */
 
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

 void _BK_gc_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	ion_reg("k", -10000.);
 	ion_reg("nca", 0.0);
 	_ca_sym = hoc_lookup("ca_ion");
 	_k_sym = hoc_lookup("k_ion");
 	_nca_sym = hoc_lookup("nca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 20, 6);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "nca_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 BK_gc BK_gc.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double, double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   acai = ( ncai ) / diff ;
   if ( acai < cai ) {
     acai = cai ;
     }
   rates ( _threadargscomma_ v , acai ) ;
   Da = ( ainf - a ) / atau ;
   Dab = ( abinf - ab ) / abtau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 acai = ( ncai ) / diff ;
 if ( acai < cai ) {
   acai = cai ;
   }
 rates ( _threadargscomma_ v , acai ) ;
 Da = Da  / (1. - dt*( ( ( ( - 1.0 ) ) ) / atau )) ;
 Dab = Dab  / (1. - dt*( ( ( ( - 1.0 ) ) ) / abtau )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   acai = ( ncai ) / diff ;
   if ( acai < cai ) {
     acai = cai ;
     }
   rates ( _threadargscomma_ v , acai ) ;
    a = a + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / atau)))*(- ( ( ( ainf ) ) / atau ) / ( ( ( ( - 1.0 ) ) ) / atau ) - a) ;
    ab = ab + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / abtau)))*(- ( ( ( abinf ) ) / abtau ) / ( ( ( ( - 1.0 ) ) ) / abtau ) - ab) ;
   }
  return 0;
}
 
double shifta ( _threadargsprotocomma_ double _lca ) {
   double _lshifta;
 _lshifta = 25.0 - 50.3 + ( 107.5 * exp ( - .12 * _lca * 1e3 ) ) ;
   
return _lshifta;
 }
 
static void _hoc_shifta(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  shifta ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double peaka ( _threadargsprotocomma_ double _lca ) {
   double _lpeaka;
 _lpeaka = 2.9 + ( 6.3 * exp ( - .36 * _lca * 1e3 ) ) ;
   
return _lpeaka;
 }
 
static void _hoc_peaka(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  peaka ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double shiftab ( _threadargsprotocomma_ double _lca ) {
   double _lshiftab;
 _lshiftab = 25.0 - 55.7 + 136.9 * exp ( - .28 * _lca * 1e3 ) ;
   
return _lshiftab;
 }
 
static void _hoc_shiftab(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  shiftab ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double peakab ( _threadargsprotocomma_ double _lca ) {
   double _lpeakab;
 _lpeakab = 13.7 + 234.0 * exp ( - .72 * _lca * 1e3 ) ;
   
return _lpeakab;
 }
 
static void _hoc_peakab(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  peakab ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double taufunc ( _threadargsprotocomma_ double _lv ) {
   double _ltaufunc;
 _ltaufunc = 1.0 / ( ( 10.0 * ( exp ( - _lv / 63.6 ) + exp ( - ( 150.0 - _lv ) / 63.6 ) ) ) - 5.2 ) ;
   if ( _ltaufunc <= 0.2 ) {
     _ltaufunc = 0.2 ;
     }
   
return _ltaufunc;
 }
 
static void _hoc_taufunc(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  taufunc ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int  rates ( _threadargsprotocomma_ double _lv , double _lc ) {
   double _lrange , _lvv , _lashift , _lbshift ;
 _lashift = - 32.0 + ( 59.2 * exp ( - .09 * _lc * 1e3 ) ) + ( 96.7 * exp ( - .47 * _lc * 1e3 ) ) ;
   ainf = 1.0 / ( 1.0 + exp ( ( _lashift - _lv ) / ( 25.0 / 1.6 ) ) ) ;
   _lvv = _lv + 100.0 - shifta ( _threadargscomma_ _lc ) ;
   atau = taufunc ( _threadargscomma_ _lvv ) ;
   _lrange = peaka ( _threadargscomma_ _lc ) - 1.0 ;
   atau = ( _lrange * ( ( atau - .2 ) / .8 ) ) + 1.0 ;
   _lbshift = - 56.449 + 104.52 * exp ( - .22964 * _lc * 1e3 ) + 295.68 * exp ( - 2.1571 * _lc * 1e3 ) ;
   abinf = 1.0 / ( 1.0 + exp ( ( _lbshift - _lv ) / ( 25.0 / 1.6 ) ) ) ;
   _lvv = _lv + 100.0 - shiftab ( _threadargscomma_ _lc ) ;
   abtau = taufunc ( _threadargscomma_ _lvv ) ;
   _lrange = peakab ( _threadargscomma_ _lc ) - base ;
   abtau = ( _lrange * ( ( abtau - .2 ) / .8 ) ) + base ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
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
  cai = _ion_cai;
  ek = _ion_ek;
  ncai = _ion_ncai;
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
  cai = _ion_cai;
  ek = _ion_ek;
  ncai = _ion_ncai;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 3, 4);
   nrn_update_ion_pointer(_nca_sym, _ppvar, 4, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  ab = ab0;
  a = a0;
 {
   rates ( _threadargscomma_ v , cai ) ;
   a = ainf ;
   ab = abinf ;
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
  cai = _ion_cai;
  ek = _ion_ek;
  ncai = _ion_ncai;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gak = gakbar * a ;
   gabk = gabkbar * ab ;
   ik = ( gabk + gak ) * ( v - ek ) ;
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
  cai = _ion_cai;
  ek = _ion_ek;
  ncai = _ion_ncai;
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
  cai = _ion_cai;
  ek = _ion_ek;
  ncai = _ion_ncai;
 {   state(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = a_columnindex;  _dlist1[0] = Da_columnindex;
 _slist1[1] = ab_columnindex;  _dlist1[1] = Dab_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "BK_gc.mod";
static const char* nmodl_file_text = 
  ": Ca-dependent K channels (BK) - alphabeta4 and alpha\n"
  ": Bin Wang, Robert Brenner, and David Jaffe - Originally written May 27, 2010\n"
  ": \n"
  ": June 1, 2010 - added double exponential function for voltage-dependent activation \n"
  ":\n"
  ": July 3, 2010 - changed voltage-dependence for the two channels based on revised data\n"
  ":\n"
  ": April 2, 2011 - adjusted parameters based on updated Bin data\n"
  "\n"
  ": Mar 2016 - added instant N-type Calcium concentration (Beining et al (2016), \"A novel comprehensive and consistent electrophysiologcal model of dentate granule cells\")\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX BK_gc\n"
  "	USEION ca READ cai\n"
  "	USEION k READ ek WRITE ik\n"
  "	USEION nca READ ncai VALENCE 0\n"
  "	RANGE gakbar, gabkbar, gak, gabk, atau, ainf, a, ab, abinf, abtau,  ik, diff, acai\n"
  "	GLOBAL base\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "	(S) = (siemens)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	diff = 100000 (1)\n"
  "	gakbar = 0.0624	(S/cm2)	: maximum permeability - alpha\n"
  "	gabkbar = 0.0156	(S/cm2)	: maximum permeability - alphabeta\n"
  "	base = 4  	(mV)	: alphabeta4 base time constant\n"
  "	\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v				(mV)\n"
  "	ek			(mV)\n"
  "	\n"
  "	ik				(mA/cm2)\n"
  "      gak			(S/cm2)\n"
  "      gabk		(S/cm2)\n"
  "      ainf	\n"
  "      atau		(ms)\n"
  "      abinf	\n"
  "      abtau		(ms)\n"
  "	  cai 			(mM)\n"
  "	  ncai		(mM)\n"
  "	  acai		(mM)\n"
  "\n"
  "}\n"
  "\n"
  "STATE {ab a} :a\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD cnexp\n"
  "	gak = gakbar*a\n"
  "	gabk = gabkbar*ab\n"
  "	ik = (gabk+gak)*(v - ek)                               :+ gak\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "	: exact when v held constant; integrates over dt step\n"
  "	\n"
  "	acai = (ncai) / diff : instantaneous calcium concentration of N-type Ca channels for BK activation modified by diffusion factor for nanodomains\n"
  "\n"
  "	if (acai < cai)\n"
  "		{acai = cai}\n"
  "		\n"
  "	rates(v,acai)				      \n"
  "	a' = (ainf-a)/atau	\n"
  "	ab' = (abinf-ab)/abtau\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v, cai)\n"
  "	a = ainf\n"
  "	ab = abinf\n"
  "}\n"
  "\n"
  ": alpha channel properties\n"
  "FUNCTION shifta(ca (mM))  {\n"
  "	shifta = 25 - 50.3 + (107.5*exp(-.12*ca*1e3))\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION peaka(ca (mM))  {\n"
  "	peaka = 2.9 + (6.3*exp(-.36*ca*1e3))\n"
  "}\n"
  "\n"
  ": alpha-beta4 channel properties\n"
  "\n"
  "\n"
  "FUNCTION shiftab(ca (mM))  {\n"
  "	shiftab = 25 - 55.7 + 136.9*exp(-.28*ca*1e3)\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION peakab(ca (mM))  {\n"
  "	peakab = 13.7 + 234*exp(-.72*ca*1e3)\n"
  "}\n"
  "\n"
  ": Double sigmoid function for tau voltage-dependence\n"
  "\n"
  "\n"
  "FUNCTION taufunc(v (mV)) {\n"
  "	 taufunc = 1 / (          (10*(exp(-v/63.6) + exp (-(150-v)/63.6)))  - 5.2                  )\n"
  "	 if (taufunc <= 0.2) {	  : stop the function between 0.2 and 1\n"
  "	    taufunc = 0.2\n"
  "	 }\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV), c (mM)) { : nc (mM), \n"
  "	  LOCAL range, vv,ashift, bshift\n"
  "\n"
  "	  : alpha model\n"
  "\n"
  "	  ashift =  -32 + (59.2*exp(-.09*c*1e3)) + (96.7*exp(-.47*c*1e3))\n"
  "	  ainf = 1/(1+exp((ashift-v)/(25/1.6)))\n"
  "\n"
  "	  vv = v + 100 - shifta(c)\n"
  "	  atau = taufunc(vv)\n"
  "	  range = peaka(c)-1\n"
  "	  atau = (range*((atau-.2)/.8)) + 1\n"
  "\n"
  "	  : alpha-beta4 model\n"
  "\n"
  "	  bshift = -56.449 + 104.52*exp(-.22964*c*1e3) + 295.68*exp(-2.1571*c*1e3)\n"
  "\n"
  "	  abinf = 1/(1+exp((bshift-v)/(25/1.6)))\n"
  "\n"
  "	  vv = v + 100 - shiftab(c)\n"
  "	  abtau = taufunc(vv)\n"
  "	  range = peakab(c)-base\n"
  "	  abtau = (range*((abtau-.2)/.8)) + base		\n"
  "\n"
  "}\n"
  ;
#endif
