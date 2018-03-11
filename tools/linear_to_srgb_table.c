#include <stdio.h>
#include <math.h>

long double linear_sample_to_srgb_reference(long double in)
{
	long double tmp;
	tmp = powl(in, 1/2.4L);
	return 1.055L * tmp - 0.055L;
}

int main()
{
	int i, rgb, last;
	long double input, reference;

	last = 0;

	for (i=252; i<=65535; i++) {
		input = i / 65535.0L;
		reference = linear_sample_to_srgb_reference(input);
		rgb = round(reference * 255);
		if (rgb != last) {
			last = rgb;
			printf("%Lf\t%Lf\n", input, reference);
		}
	}
	return 0;
}
