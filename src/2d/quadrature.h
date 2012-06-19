////////////////////////////////////////
// 2D Quadrature Rules
////////////////////////////////////////

// order goes (r1, s1, w1, r2, s2, w2, ...)
// 1 point
double quad_2d_degree1[] = {0.333333333333333, 0.333333333333333, 1.0};
// 3 points
double quad_2d_degree2[] = {0.166666666666666, 0.166666666666666, 0.333333333333333,
                           0.666666666666666, 0.166666666666666, 0.333333333333333,
                           0.166666666666666, 0.666666666666666, 0.333333333333333};
// 4 points
double quad_2d_degree3[] = {0.333333333333333,0.3333333333333333,-0.5625,
                           0.6,0.2,.520833333333333,
                           0.2,0.6,.520833333333333,
                           0.2,0.2,.520833333333333};
// 6 points
double quad_2d_degree4[] = {0.816847572980459,0.091576213509771,0.109951743655322,
                           0.091576213509771,0.816847572980459,0.109951743655322,
                           0.091576213509771,0.091576213509771,0.109951743655322,
                           0.108103018168070,0.445948490915965,0.223381589678011,
                           0.445948490915965,0.108103018168070,0.223381589678011,
                           0.445948490915965,0.445948490915965,0.223381589678011};
// 7 points
double quad_2d_degree5[] = {0.333333333333333,0.333333333333333,0.225000000000000,
                           0.797426985353087,0.101286507323456,0.125939180544827,
                           0.101286507323456,0.797426985353087,0.125939180544827,
                           0.101286507323456,0.101286507323456,0.125939180544827,
                           0.470142064105115,0.059715871789770,0.132394152788506,
                           0.059715871789770,0.470142064105115,0.132394152788506,
                           0.470142064105115,0.470142064105115,0.132394152788506};
// 12 points
double quad_2d_degree6[] = {0.873821971016996,0.063089014491502,0.050844906370207,
                           0.063089014491502,0.873821971016996,0.050844906370207,
                           0.063089014491502,0.063089014491502,0.050844906370207,
                           0.501426509658179,0.249286745170910,0.116786275726379,
                           0.249286745170910,0.501426509658179,0.116786275726379,
                           0.249286745170910,0.249286745170910,0.116786275726379,
                           0.636502499121399,0.310352451033784,0.082851075618374,
                           0.310352451033784,0.636502499121399,0.082851075618374,
                           0.636502499121399,0.053145049844816,0.082851075618374,
                           0.310352451033784,0.053145049844816,0.082851075618374,
                           0.053145049844816,0.310352451033785,0.082851075618374,
                           0.053145049844816,0.636502499121399,0.082851075618374};
// 13 points
double quad_2d_degree7[] = {0.333333333333333,0.333333333333333,-0.149570044467682,
                           0.479308067841920,0.260345966079040,0.175615257433208,
                           0.260345966079040,0.479308067841920,0.175615257433208,
                           0.260345966079040,0.260345966079040,0.175615257433208,
                           0.869739794195568,0.065130102902216,0.053347235608838,
                           0.065130102902216,0.869739794195568,0.053347235608838,
                           0.065130102902216,0.065130102902216,0.053347235608838,
                           0.048690315425316,0.312865496004874,0.077113760890257,
                           0.312865496004874,0.048690315425316,0.077113760890257,
                           0.638444188569810,0.048690315425316,0.077113760890257,
                           0.048690315425316,0.638444188569810,0.077113760890257,
                           0.312865496004874,0.638444188569810,0.077113760890257,
                           0.638444188569810,0.312865496004874,0.077113760890257};
// 16 points
double quad_2d_degree8[] = {0.333333333333333,0.333333333333333,0.144315607677787,
                           0.081414823414554,0.459292588292723,0.095091634267285,
                           0.459292588292723,0.081414823414554,0.095091634267285,
                           0.459292588292723,0.459292588292723,0.095091634267285,
                           0.658861384496480,0.170569307751760,0.103217370534718,
                           0.170569307751760,0.658861384496480,0.103217370534718,
                           0.170569307751760,0.170569307751760,0.103217370534718,
                           0.898905543365938,0.050547228317031,0.032458497623198,
                           0.050547228317031,0.898905543365938,0.032458497623198,
                           0.050547228317031,0.050547228317031,0.032458497623198,  
                           0.008394777409958,0.728492392955404,0.027230314174435,
                           0.728492392955404,0.008394777409958,0.027230314174435,
                           0.263112829634638,0.008394777409958,0.027230314174435,
                           0.008394777409958,0.263112829634638,0.027230314174435,
                           0.263112829634638,0.728492392955404,0.027230314174435,
                           0.728492392955404,0.263112829634638,0.027230314174435};
// 19 points
double quad_2d_degree9[] = {0.333333333333333,0.333333333333333,0.097135796282799,
                           0.020634961602525,0.489682519198738,0.031334700227139,
                           0.489682519198738,0.020634961602525,0.031334700227139,
                           0.489682519198738,0.489682519198738,0.031334700227139,
                           0.125820817014127,0.437089591492937,0.077827541004774,
                           0.437089591492937,0.125820817014127,0.077827541004774,
                           0.437089591492937,0.437089591492937,0.077827541004774,
                           0.623592928761935,0.188203535619033,0.079647738927210,
                           0.188203535619033,0.623592928761935,0.079647738927210,
                           0.188203535619033,0.188203535619033,0.079647738927210,
                           0.910540973211095,0.044729513394453,0.025577675658698,
                           0.044729513394453,0.910540973211095,0.025577675658698,
                           0.044729513394453,0.044729513394453,0.025577675658698,
                           0.036838412054736,0.221962989160766,0.043283539377289,
                           0.221962989160766,0.036838412054736,0.043283539377289,
                           0.036838412054736,0.741198598784498,0.043283539377289,
                           0.741198598784498,0.036838412054736,0.043283539377289,
                           0.741198598784498,0.221962989160766,0.043283539377289,
                           0.221962989160766,0.741198598784498,0.043283539377289};
// 25 points
double quad_2d_degree10[] = {0.333333333333333,0.333333333333333,0.090817990382754,
                           0.028844733232685,0.485577633383657,0.036725957756467,
                           0.485577633383657,0.028844733232685,0.036725957756467,
                           0.485577633383657,0.485577633383657,0.036725957756467,
                           0.781036849029926,0.109481575485037,0.045321059435528,
                           0.109481575485037,0.781036849029926,0.045321059435528,
                           0.109481575485037,0.109481575485037,0.045321059435528,
                           0.141707219414880,0.307939838764121,0.072757916845420,
                           0.307939838764121,0.141707219414880,0.072757916845420,
                           0.307939838764121,0.550352941820999,0.072757916845420,
                           0.550352941820999,0.307939838764121,0.072757916845420,
                           0.550352941820999,0.141707219414880,0.072757916845420,
                           0.141707219414880,0.550352941820999,0.072757916845420,
                           0.025003534762686,0.246672560639903,0.028327242531057,
                           0.246672560639903,0.025003534762686,0.028327242531057,
                           0.025003534762686,0.728323904597411,0.028327242531057,
                           0.728323904597411,0.025003534762686,0.028327242531057,
                           0.728323904597411,0.246672560639903,0.028327242531057,
                           0.246672560639903,0.728323904597411,0.028327242531057,
                           0.009540815400299,0.066803251012200,0.009421666963733,
                           0.066803251012200,0.009540815400299,0.009421666963733,
                           0.066803251012200,0.923655933587500,0.009421666963733,
                           0.923655933587500,0.066803251012200,0.009421666963733,
                           0.923655933587500,0.009540815400299,0.009421666963733,
                           0.009540815400299,0.923655933587500,0.009421666963733};
// put them together 
double *quad_2d[] = {quad_2d_degree1, quad_2d_degree2, quad_2d_degree3,
                     quad_2d_degree4, quad_2d_degree5, quad_2d_degree6,
                     quad_2d_degree7, quad_2d_degree8, quad_2d_degree9,
                     quad_2d_degree10}; 

////////////////////////////////////////
// 1D Quadrature Rules
////////////////////////////////////////

// order goes (r1, w1, r2, w2, ...)
double quad_1d_degree1[] = {0.0,2.0};
double quad_1d_degree2[] = {-1./sqrt(3),1.,
                             1./sqrt(3),1.};
double quad_1d_degree3[] = {-sqrt(3./5), 5./9, 
                           0., 8./9,
                           sqrt(3./5), 5./9};

double quad_1d_degree4[] = {-sqrt((3.+2.*sqrt(6./5))/7.), (18.-sqrt(30.))/36.,
                           -sqrt((3.-2.*sqrt(6./5))/7.), (18.+sqrt(30.))/36.,
                           sqrt((3.-2.*sqrt(6./5))/7.), (18.+sqrt(30.))/36.,
                           sqrt((3.+2.*sqrt(6./5))/7.), (18.-sqrt(30.))/36.};
double quad_1d_degree5[] = {-sqrt(5.+2.*sqrt(10./7))/3., (322.-13.*sqrt(70.))/900.,
                           -sqrt(5.-2.*sqrt(10./7))/3., (322.+13.*sqrt(70.))/900.,
                           0., 128./225,
                           sqrt(5.-2.*sqrt(10./7))/3., (322.+13.*sqrt(70.))/900.,
                           sqrt(5.+2.*sqrt(10./7))/3., (322.-13.*sqrt(70.))/900.};
double quad_1d_degree6[] = {-0.93246951, 0.17132449,
                           -0.66120939, 0.36076157,
                           -0.23861918, 0.46791393,
                           0.23861918, 0.46791393,
                           0.66120939, 0.36076157,
                           0.93246951, 0.17132449};
double quad_1d_degree7[] = {-0.94910791, 0.12948497,
                           -0.74153119, 0.27970539,
                           -0.40584515, 0.38183005,
                            0., 0.41795918,
                           0.40584515, 0.38183005,
                           0.74153119, 0.27970539,
                           0.94910791, 0.12948497};
double quad_1d_degree8[] = {-0.18343464, 0.36268378,
                           -0.96028986, 0.10122854,
                           -0.79666648, 0.22238103,
                           -0.52553241, 0.31370665,
                           0.18343464, 0.36268378,
                           0.52553241, 0.31370665,
                           0.79666648, 0.22238103,
                           0.96028986, 0.10122854};

double *quad_1d[] = {quad_1d_degree1, quad_1d_degree2, quad_1d_degree3,
                    quad_1d_degree4, quad_1d_degree5, quad_1d_degree6,
                    quad_1d_degree7, quad_1d_degree8};
