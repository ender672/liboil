#ifndef OIL_RESAMPLE_INTERNAL_H
#define OIL_RESAMPLE_INTERNAL_H

/* Lookup tables shared between oil_resample.c and arch-specific files. */
extern float s2l_map[256];
extern float i2f_map[256];
extern unsigned char *l2s_map;
extern int l2s_len;

#endif
