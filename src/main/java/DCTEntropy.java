

import org.jtransforms.dct.DoubleDCT_2D;

public class DCTEntropy {
	
	protected final int mWidth;
	protected final int mHeight;
	protected final double[] array;
	
	protected DCTEntropy()
	{
		array = new double[0];
		mWidth = 0;
		mHeight = 0;
	}
	
	public DCTEntropy(  final int pWidth,
						final int pHeight,
						final double[] pArray)
	{
		super();
		array = pArray;
		mWidth = pWidth;
		mHeight = pHeight;
	}
	public boolean copyFrom(final int pWidth,
							final int pHeight,
							final double[] pArray)
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
	
	public final double normL2()
	{
		final int length = array.length;
		final double[] marray = array;
		double norm = 0;
		for (int i = 0; i < length; i++)
		{
			final double value = marray[i];
			norm += value * value;
		}
		norm = (double)Math.sqrt(norm);
		return norm;
	}	
	
	public final double normalizeNormL2()
	{
		final double norm = normL2();
		if (norm != 0)
		{
			final double invnorm = 1 / norm;

			final int length = array.length;
			final double[] marray = array;
			for (int i = 0; i < length; i++)
			{
				final double value = marray[i];
				marray[i] = value * invnorm;
			}
		}
		return norm;
	}
	
	public final void dctforward()
	{
		final DoubleDCT_2D dct = new DoubleDCT_2D(mHeight, mWidth);
		dct.forward(array,true);
	}	
	
	public final double entropyShannonSubTriangle(	final int xl,
													final int yl,
													final int xh,
													final int yh)
	{
		final int width = mWidth;
		final double[] marray = array;
		double entropy = 0;
		for (int y = yl; y < yh; y++)
		{
			final int yi = y * width;
			final int xend = xh - y * xh / yh;
			for (int x = xl; x < xend; x++)
			{
				final int i = yi + x;
				final double value = marray[i];
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
	
	public final double compute(final double pPSFSupportDiameter) 
	{
		dctforward();
		normalizeNormL2();
		final int lOTFSupportX = (int) (mWidth / pPSFSupportDiameter)+1;
		final int lOTFSupportY = (int) (mHeight / pPSFSupportDiameter)+1;		
		final double dEntropy = entropyShannonSubTriangle(  0,
															0,
															lOTFSupportX,
															lOTFSupportY);
		return dEntropy;
	}
	
} 