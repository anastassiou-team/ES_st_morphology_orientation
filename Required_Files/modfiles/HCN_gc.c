/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
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
 
#define nrn_init _nrn_init__HCN
#define _nrn_initial _nrn_initial__HCN
#define nrn_cur _nrn_cur__HCN
#define _nrn_current _nrn_current__HCN
#define nrn_jacob _nrn_jacob__HCN
#define nrn_state _nrn_state__HCN
#define _net_receive _net_receive__HCN 
#define rate rate__HCN 
#define states states__HCN 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gbar _p[0]
#define gbar_columnindex 0
#define i _p[1]
#define i_columnindex 1
#define linf _p[2]
#define linf_columnindex 2
#define taul _p[3]
#define taul_columnindex 3
#define g _p[4]
#define g_columnindex 4
#define l1 _p[5]
#define l1_columnindex 5
#define l2 _p[6]
#define l2_columnindex 6
#define Dl1 _p[7]
#define Dl1_columnindex 7
#define Dl2 _p[8]
#define Dl2_columnindex 8
#define _g _p[9]
#define _g_columnindex 9
 
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
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rate(void);
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
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_HCN", _hoc_setdata,
 "rate_HCN", _hoc_rate,
 0, 0
};
 /* declare global and static user variables */
#define Ac Ac_HCN
 double Ac = 14;
#define at at_HCN
 double at = 0.00052;
#define bt bt_HCN
 double bt = 0.2151;
#define chalf chalf_HCN
 double chalf = -0.8;
#define cAMP cAMP_HCN
 double cAMP = 0;
#define e e_HCN
 double e = -41.9;
#define kc kc_HCN
 double kc = -0.35;
#define kl kl_HCN
 double kl = 8;
#define qt qt_HCN
 double qt = 0;
#define qfact qfact_HCN
 double qfact = 1;
#define q10 q10_HCN
 double q10 = 1;
#define vhalft vhalft_HCN
 double vhalft = 30.4;
#define vhalfl vhalfl_HCN
 double vhalfl = -75.3;
#define vhalfc vhalfc_HCN
 double vhalfc = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "e_HCN", "mV",
 "vhalfl_HCN", "mV",
 "vhalft_HCN", "mV",
 "at_HCN", "/ms",
 "bt_HCN", "/ms",
 "Ac_HCN", "mV",
 "kc_HCN", "/uM",
 "chalf_HCN", "uM",
 "cAMP_HCN", "uM",
 "vhalfc_HCN", "mV",
 "gbar_HCN", "S/cm2",
 "i_HCN", "mA/cm2",
 "g_HCN", "S/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double l20 = 0;
 static double l10 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "e_HCN", &e_HCN,
 "vhalfl_HCN", &vhalfl_HCN,
 "kl_HCN", &kl_HCN,
 "vhalft_HCN", &vhalft_HCN,
 "at_HCN", &at_HCN,
 "bt_HCN", &bt_HCN,
 "q10_HCN", &q10_HCN,
 "qfact_HCN", &qfact_HCN,
 "Ac_HCN", &Ac_HCN,
 "kc_HCN", &kc_HCN,
 "chalf_HCN", &chalf_HCN,
 "cAMP_HCN", &cAMP_HCN,
 "vhalfc_HCN", &vhalfc_HCN,
 "qt_HCN", &qt_HCN,
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
 
#define _cvode_ieq _ppvar[0]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"HCN",
 "gbar_HCN",
 0,
 "i_HCN",
 "linf_HCN",
 "taul_HCN",
 "g_HCN",
 0,
 "l1_HCN",
 "l2_HCN",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 10, _prop);
 	/*initialize range parameters*/
 	gbar = 4e-06;
 	_prop->param = _p;
 	_prop->param_size = 10;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _HCN_gc_reg() {
	int _vectorized = 0;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 10, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 HCN HCN_gc.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "I-h channel ";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rate(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rate ( _threadargscomma_ v ) ;
   Dl1 = ( linf - l1 ) / taul ;
   Dl2 = ( linf - l2 ) / ( taul * 6.4 ) ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rate ( _threadargscomma_ v ) ;
 Dl1 = Dl1  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taul )) ;
 Dl2 = Dl2  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ( taul * 6.4 ) )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rate ( _threadargscomma_ v ) ;
    l1 = l1 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taul)))*(- ( ( ( linf ) ) / taul ) / ( ( ( ( - 1.0 ) ) ) / taul ) - l1) ;
    l2 = l2 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ( taul * 6.4 ))))*(- ( ( ( linf ) ) / ( taul * 6.4 ) ) / ( ( ( ( - 1.0 ) ) ) / ( taul * 6.4 ) ) - l2) ;
   }
  return 0;
}
 
static int  rate (  double _lv ) {
   if ( cAMP <= 0.0 ) {
     vhalfc = 0.0 ;
     }
   else {
     vhalfc = Ac / ( 1.0 + exp ( ( log10 ( cAMP ) - chalf ) / kc ) ) ;
     }
   linf = 1.0 / ( 1.0 + exp ( ( _lv - vhalfl - vhalfc ) / kl ) ) ;
   taul = 1.0 / ( qt * qfact * ( at * exp ( - _lv / vhalft ) + bt * exp ( _lv / vhalft ) ) ) ;
    return 0; }
 
static void _hoc_rate(void) {
  double _r;
   _r = 1.;
 rate (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 ();
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  l2 = l20;
  l1 = l10;
 {
   qt = pow( q10 , ( ( celsius - 33.0 ) / 10.0 ) ) ;
   rate ( _threadargscomma_ v ) ;
   l1 = linf ;
   l2 = linf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = gbar * ( 0.8 * l1 + 0.2 * l2 ) ;
   i = g * ( v - e ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 { error =  states();
 if(error){fprintf(stderr,"at line 83 in file HCN_gc.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 }}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = l1_columnindex;  _dlist1[0] = Dl1_columnindex;
 _slist1[1] = l2_columnindex;  _dlist1[1] = Dl2_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "HCN_gc.mod";
static const char* nmodl_file_text = 
  "TITLE I-h channel \n"
  "\n"
  "COMMENT\n"
  "\n"
  "Start from  Magee 1998 for distal dendrites\n"
  "Adaped from Li & Ascoli 2006,2008 \n"
  "Adapted by A. Hanuschkin 2011\n"
  "Adapted by M. Beining 2016 (added slow component and cAMP dependence with data  from Chen et al 2001 Journal of General Physiology)\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX HCN\n"
  "	NONSPECIFIC_CURRENT i\n"
  "    RANGE gbar, g, taul, linf\n"
  "    GLOBAL vhalfc, e, vhalfl, kl, vhalft, at, bt, q10, qfact, cAMP, qt\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "    (S)  = (siemens)\n"
  "	(molar) = (1/liter)\n"
  "	(uM) = (micromolar)		\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v 		(mV)\n"
  "    e = -41.9	(mV)    : exp data		(see manuscript)\n"
  "	gbar=4e-06 	(S/cm2) : dummy default\n"
  "\n"
  "				: Boltzman fit to steady state currents\n"
  "        vhalfl=-75.3	(mV)	: fitted Boltzmann	(see manuscript) \n"
  "	kl=8			: fitted Boltzmann      (see manuscript) \n"
  "        \n"
  ": Fit to activation/deactivation time constance\n"
  ": own fit\n"
  ": E               = 0.000392656      +/- 0.0001422    (36.22%)\n"
  ": F               = 0.395751         +/- 0.1126       (28.45%)\n"
  ": G               = 29.385           +/- 2.715        (9.238%)\n"
  ": manuscript fast, a = 0.52 +/- 0.21 ms, b = 215.1 +/- 65.7 ms, V0 = 30.4 +/- 3.4 mV, \n"
  ": E               = 0.00052\n"
  ": F               = 0.2151        \n"
  ": G               = 30.4      \n"
  " \n"
  "    vhalft=30.4	 (mV)    : fitted 		(see manuscript)\n"
  "    at=0.00052	 (/ms)   : fitted 		(see manuscript)\n"
  "	bt=0.2151	 (/ms)	 :fitted               (see manuscript)\n"
  "\n"
  "				: Temperature dependence\n"
  "    celsius         (degC)  : unused\n"
  "	q10=1.			: no correction for Temperature via q10 to save computation time (uncomment in rate to make T dep simulations... \n"
  "	qfact = 1 \n"
  "	\n"
  "	Ac = 14 (mV)\n"
  "	kc 	= -0.35 (/uM)\n"
  "	chalf 	= -0.8 (uM)\n"
  "	cAMP = 0 (uM)\n"
  "	\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	i (mA/cm2)\n"
  "        linf      \n"
  "        taul\n"
  "        g (S/cm2)\n"
  "		vhalfc (mV)  : voltage shift by cAMP (for HCN1+HCN2 heteromers)\n"
  "		qt\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	qt = q10^((celsius-33)/10)\n"
  "	rate(v)\n"
  "	l1=linf\n"
  "	l2=linf\n"
  "}\n"
  "\n"
  "STATE { l1 l2 }\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	g = gbar*(0.8*l1 + 0.2*l2)  : slow component has contribution of ~ 20 % to total current\n"
  "	i = g*(v-e)\n"
  "\n"
  "}\n"
  "\n"
  "DERIVATIVE states {     : exact when v held constant; integrates over dt step\n"
  "        rate(v)\n"
  "        l1' =  (linf - l1)/taul\n"
  "		l2' = (linf - l2)/(taul*6.4)  : slow component has same time course as fast one but is slower by factor of ~6.4 (see Stegen&Hanuschkin 2012 Figure 3)\n"
  "}\n"
  "\n"
  "PROCEDURE rate(v (mV)) { :callable from hoc\n"
  "        :LOCAL qt\n"
  "\n"
  "	:qt = 1 	: saves computational time...\n"
  "	\n"
  "	if (cAMP <= 0) {\n"
  "		vhalfc = 0\n"
  "	}else{\n"
  "		vhalfc = Ac/(1 + exp((log10(cAMP)-chalf)/kc))  : from Chen et al 2001 Journal of General Physiology\n"
  "	}\n"
  "    linf = 1/(1 + exp((v-vhalfl-vhalfc)/kl))\n"
  " 	taul = 1/(qt * qfact * (at*exp(-v/vhalft) + bt*exp(v/vhalft) ))\n"
  "}\n"
  ;
#endif
