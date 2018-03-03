#include <stdio.h>
#include <math.h>

static int srgb_sample_to_linear_reference(int in)
{
	long double in_f, result, tmp;
	in_f = in / 255.0L;
	if (in_f <= 0.0404482362771082L) {
		result = in_f / 12.92L;
	} else {
		tmp = ((in_f + 0.055L)/1.055L);
		result = powl(tmp, 2.4L);
	}
	return round(result * 65535);
}

int main()
{
	int input, reference;

	for (input=0; input<=255; input++) {
		reference = srgb_sample_to_linear_reference(input);
		printf("0x%04x", reference);
		if (input % 8 == 7) {
			printf(",\n");
		} else {
			printf(", ");
		}
	}
	return 0;
}
