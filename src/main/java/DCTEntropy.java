

import edu.emory.mathcs.jtransforms.dct.FloatDCT_2D;

public class DCTEntropy {
	
	protected final int mWidth;
	protected final int mHeight;
	protected final float[] array;
	
	protected DCTEntropy()
	{
		array = new float[0];
		mWidth = 0;
		mHeight = 0;
	}
	
	public DCTEntropy(  final int pWidth,
						final int pHeight,
						final float[] pArray)
	{
		super();
		array = pArray;
		mWidth = pWidth;
		mHeight = pHeight;
	}
	public boolean copyFrom(final int pWidth,
							final int pHeight,
							final float[] pArray)
	{
		if (mWidth == pWidth && mHeight == pHeight)
		{
			System.arraycopy(	pArray,
								0,
								array,
								0,
								mWidth * mHeight);
			return true;
		}
		else
		{
			return false;
		}

	}
	
	public final float normL2()
	{
		final int length = array.length;
		final float[] marray = array;
		float norm = 0;
		for (int i = 0; i < length; i++)
		{
			final float value = marray[i];
			norm += value * value;
		}
		norm = (float)Math.sqrt(norm);
		return norm;
	}	
	
	public final float normalizeNormL2()
	{
		final float norm = normL2();
		if (norm != 0)
		{
			final float invnorm = 1 / norm;

			final int length = array.length;
			final float[] marray = array;
			for (int i = 0; i < length; i++)
			{
				final float value = marray[i];
				marray[i] = value * invnorm;
			}
		}
		return norm;
	}
	
	public final void dctforward()
	{
		final FloatDCT_2D dct = new FloatDCT_2D(mHeight, mWidth);
		dct.forward(array,false);
	}	
	
	public final float entropyShannonSubTriangle(	final int xl,
													final int yl,
													final int xh,
													final int yh)
	{
		final int width = mWidth;
		final float[] marray = array;
		float entropy = 0;
		for (int y = yl; y < yh; y++)
		{
			final int yi = y * width;
			final int xend = xh - y * xh / yh;
			for (int x = xl; x < xend; x++)
			{
				final int i = yi + x;
				final float value = marray[i];
				if (value > 0)
				{
					entropy += value * Math.log(value);
				}
				else if (value < 0)
				{
					entropy += -value * Math.log(-value);
				}
		
			}
		}
		entropy = -entropy;
		entropy = 2 * entropy / ((xh - xl) * (yh - yl));
		return entropy;
	}
	
	public final float compute(final float pPSFSupportDiameter) 
	{
		dctforward();
		normalizeNormL2();
		final int lOTFSupportX = (int) (mWidth / pPSFSupportDiameter);
		final int lOTFSupportY = (int) (mHeight / pPSFSupportDiameter);		
		final float dEntropy = entropyShannonSubTriangle(  0,
															0,
															lOTFSupportX,
															lOTFSupportY);
		return dEntropy;
	}
	
} 