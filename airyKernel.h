extern "C" {

struct GimpExport {
  unsigned int width;
  unsigned int height;
  unsigned int bytes_per_pixel; /* 2:RGB16, 3:RGB, 4:RGBA */
  unsigned char pixel_data[17 * 17 * 4 + 1];
};

static const GimpExport airyKernel {
17, 17, 4,
  "\335\315\315\004\323\272\265\005\330\352\334\005\324\340\322\006\312\253\247\010\306"
  "\240\242\013\317\275\270\013\316\314\301\013\317\323\306\013\316\314\301\013\317"
  "\275\270\013\306\240\242\013\312\253\247\010\324\340\322\006\330\352\334\005\323"
  "\272\265\005\335\315\315\004\323\272\265\005\327\356\337\005\324\316\303\006\304\233"
  "\235\012\322\320\304\013\313\351\317\012\307\336\312\013\304\303\274\014\301\271"
  "\270\015\304\303\274\014\307\336\312\013\313\351\317\012\322\320\304\013\304\233"
  "\235\012\324\316\303\006\327\356\337\005\323\272\265\005\331\351\334\005\324\316\304"
  "\006\307\242\244\013\321\343\316\012\306\331\307\013\301\234\252\020\311\245\252"
  "\025\317\275\263\026\322\311\271\026\317\275\263\026\311\245\252\025\301\234\252"
  "\020\306\331\307\013\321\343\316\012\307\242\244\013\324\316\304\006\331\351\334"
  "\005\326\343\325\006\303\232\234\012\321\343\316\012\304\307\276\014\305\234\246"
  "\024\322\326\276\026\302\341\313\026\252\277\302\030\255\270\303\030\252\277\302"
  "\030\302\341\313\026\322\326\276\026\305\234\246\024\304\307\276\014\321\343\316"
  "\012\303\232\234\012\326\343\325\006\311\254\250\010\322\317\304\013\306\331\307"
  "\013\304\233\246\024\321\343\306\026\251\267\302\030\266\217\245\"\322\270\256"
  ")\330\306\264+\322\270\256)\266\217\245\"\251\267\302\030\321\343\306\026\304"
  "\233\246\024\306\331\307\013\322\317\304\013\311\254\250\010\304\234\237\013\314"
  "\352\320\012\302\236\254\020\322\325\276\026\251\267\301\030\307\246\247'\335"
  "\345\317.\265\314\317\062\241\265\306\063\265\314\317\062\335\345\317.\307\246"
  "\247'\251\267\301\030\322\325\276\026\302\236\254\020\314\352\320\012\304\234"
  "\237\013\316\271\266\013\307\341\312\012\307\240\247\025\304\343\313\026\264\216"
  "\246\"\335\343\316.\233\243\275\065\311\257\254L\335\313\275_\311\257\254"
  "L\233\243\275\065\335\343\316.\264\216\246\"\304\343\313\026\307\240\247\025"
  "\307\341\312\012\316\271\266\013\320\313\300\013\306\312\301\013\317\273\262"
  "\026\253\303\302\027\321\264\255(\270\317\320\062\307\255\253K\363\361\352\220"
  "\356\362\363\270\363\361\352\220\307\255\253K\270\317\320\062\321\264\255"
  "(\253\303\302\027\317\273\262\026\306\312\301\013\320\313\300\013\316\316\302"
  "\013\301\275\270\014\320\303\266\026\251\271\301\030\327\302\263*\245\274\311"
  "\063\333\307\271[\357\362\363\264\327\336\347\363\357\362\363\264\333\307"
  "\271[\245\274\311\063\327\302\263*\251\271\301\030\320\303\266\026\301\275\270"
  "\014\316\316\302\013\317\311\277\013\306\313\301\013\316\271\261\026\254\306\303"
  "\027\317\261\254(\274\323\322\061\303\250\251I\363\360\347\213\361\363\363"
  "\260\363\360\347\213\303\250\251I\274\323\322\061\317\261\254(\254\306\303"
  "\027\316\271\261\026\306\313\301\013\317\311\277\013\316\267\265\013\307\342\313"
  "\012\306\236\246\025\307\344\313\026\262\214\247!\336\342\313-\234\252\301\064"
  "\300\245\250H\331\303\267W\300\245\250H\234\252\301\064\336\342\313-\262\214"
  "\247!\307\344\313\026\306\236\246\025\307\342\313\012\316\267\265\013\304\234"
  "\236\012\314\350\317\012\301\241\255\017\322\321\274\026\254\277\303\030\303\237"
  "\245&\336\341\312-\300\326\323\061\251\301\313\063\300\326\323\061\336\341\312"
  "-\303\237\245&\254\277\303\030\322\321\274\026\301\241\255\017\314\350\317\012"
  "\304\234\236\012\313\261\254\010\321\310\300\013\306\335\310\013\303\230\245"
  "\024\322\337\304\026\254\300\304\030\261\215\250\040\315\255\253'\325\276\261"
  ")\315\255\253'\261\215\250\040\254\300\304\030\322\337\304\026\303\230\245\024"
  "\306\335\310\013\321\310\300\013\313\261\254\010\325\346\327\005\304\234\235\012"
  "\322\340\316\012\306\320\303\013\303\230\245\023\323\320\274\026\310\344\313"
  "\026\255\311\304\027\247\273\300\030\255\311\304\027\310\344\313\026\323\320\274"
  "\026\303\230\245\023\306\320\303\013\322\340\316\012\304\234\235\012\325\346\327"
  "\005\331\347\332\005\323\322\306\006\305\235\237\013\322\337\316\012\306\337\311"
  "\013\302\243\256\017\305\234\245\024\316\266\260\026\316\275\262\026\316\266\260"
  "\026\305\234\245\024\302\243\256\017\306\337\311\013\322\337\316\012\305\235\237"
  "\013\323\322\306\006\331\347\332\005\317\264\260\005\331\353\335\005\323\324\310\006"
  "\304\234\235\012\320\306\277\013\315\347\317\012\310\345\315\012\307\321\304"
  "\013\306\307\277\014\307\321\304\013\310\345\315\012\315\347\317\012\320\306\277"
  "\013\304\234\235\012\323\324\310\006\331\353\335\005\317\264\260\005\335\320\317"
  "\004\320\265\261\005\332\347\332\005\325\346\327\005\314\264\256\010\303\232\235\012"
  "\315\265\263\013\320\307\276\013\320\314\300\013\320\307\276\013\315\265\263"
  "\013\303\232\235\012\314\264\256\010\325\346\327\005\332\347\332\005\320\265\261"
  "\005\335\320\317\004",
};

//   15, 15, 4,
//   "\372\267\265\004\357\302\241\005\241\372\302\005\350\270\226\007\377\206\212\012\352"
//   "\263\233\013\314\322\233\013\276\333\232\013\314\322\233\013\352\263\233\013\377"
//   "\206\212\012\350\270\226\007\241\372\302\005\357\302\241\005\372\267\265\004\360\301"
//   "\240\005\253\365\274\005\373\216\204\011\345\274\240\013\207\371\225\012\263\324"
//   "\243\013\340\241\255\016\352\224\262\017\340\241\255\016\263\324\243\013\207\371"
//   "\225\012\345\274\240\013\373\216\204\011\253\365\274\005\360\301\240\005\241\373"
//   "\302\005\373\216\204\011\323\324\243\013\234\344\233\013\371}\257\021\354\263\207"
//   "\026\302\336o\026\262\353w\026\302\336o\026\354\263\207\026\371}\257\021\234\344"
//   "\233\013\323\324\243\013\373\216\204\011\241\373\302\005\346\273\227\007\346\273"
//   "\241\013\232\346\234\013\374\203\243\024\260\352x\026\204\305\321\030\324\202"
//   "\323\036\343x\302!\324\202\323\036\204\305\321\030\260\352x\026\374\203\243\024"
//   "\232\346\234\013\346\273\241\013\346\273\227\007\376\206\212\012\211\372\227\012"
//   "\371}\257\021\263\351w\026\270\226\334\033\370\256})\312\347\234.\253\347\277"
//   "/\312\347\234.\370\256})\270\226\334\033\263\351w\026\371}\257\021\211\372\227"
//   "\012\376\206\212\012\355\257\234\013\255\330\242\013\357\260\212\026\200\310\320"
//   "\030\371\254~)\210\335\330\061\311\211\306>\354\240\233L\311\211\306>\210\335"
//   "\330\061\371\254~)\200\310\320\030\357\260\212\026\255\330\242\013\355\257\234"
//   "\013\321\317\235\013\334\247\255\016\307\332o\026\317\206\325\035\316\346\227"
//   ".\306\211\311=\374\346\272\200\355\363\357\252\374\346\272\200\306\211\311"
//   "=\316\346\227.\317\206\325\035\307\332o\026\334\247\255\016\321\317\235\013\310"
//   "\327\226\013\344\231\257\017\267\345r\026\340z\311\040\263\350\270/\347\233\237"
//   "J\357\363\354\247\311\341\377\357\357\363\354\247\347\233\237J\263\350\270"
//   "/\340z\311\040\267\345r\026\344\231\257\017\310\327\226\013\322\315\235\013\332"
//   "\252\255\016\311\330p\026\314\211\326\035\321\344\222.\300\213\320<\375\342"
//   "\261{\361\363\351\243\375\342\261{\300\213\320<\321\344\222.\314\211\326"
//   "\035\311\330p\026\332\252\255\016\322\315\235\013\360\254\231\013\250\335\240"
//   "\013\362\252\215\026y\316\313\030\371\245\206(\224\341\317\060\274\214\323;\342"
//   "\225\243H\274\214\323;\224\341\317\060\371\245\206(y\316\313\030\362\252\215"
//   "\026\250\335\240\013\360\254\231\013\376\204\207\012\217\366\227\012\366\201\260"
//   "\021\275\346t\026\254\241\335\032\371\243\210(\325\342\216-\272\351\260/\325"
//   "\342\216-\371\243\210(\254\241\335\032\275\346t\026\366\201\260\021\217\366"
//   "\227\012\376\204\207\012\341\302\232\007\353\263\235\013\221\355\230\013\374~\247"
//   "\023\276\345t\026w\321\311\027\310\215\330\034\333\177\317\037\310\215\330\034"
//   "w\321\311\027\276\345t\026\374~\247\023\221\355\230\013\353\263\235\013\341\302"
//   "\232\007\244\372\301\005\370\223\205\010\331\312\241\013\220\355\230\013\366\202"
//   "\262\020\364\247\217\026\316\324r\026\275\337o\026\316\324r\026\364\247\217\026"
//   "\366\202\262\020\220\355\230\013\331\312\241\013\370\223\205\010\244\372\301"
//   "\005\362\273\240\005\240\371\274\005\370\224\206\010\354\262\235\013\222\365\226"
//   "\012\242\341\240\013\326\260\253\015\340\237\256\016\326\260\253\015\242\341\240"
//   "\013\222\365\226\012\354\262\235\013\370\224\206\010\240\371\274\005\362\273\240"
//   "\005\371\302\277\004\363\272\242\005\246\373\301\005\337\304\231\007\376\204\207\012"
//   "\363\250\232\013\330\307\237\013\315\325\231\013\330\307\237\013\363\250\232"
//   "\013\376\204\207\012\337\304\231\007\246\373\301\005\363\272\242\005\371\302\277"
//   "\004",
// };

}