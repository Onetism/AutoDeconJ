

/*
 * @Author: Onetism
 * @Date: 2020-11-26 18:55:12
 * @LastEditTime: 2022-08-21 09:48:08
 * @LastEditors: onetism onetism@163.com
 * @Description: In User Settings Edit
 * @FilePath: \Auto_LF_Deconvolution\src\test.java
 */
import ij.*;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;


public class App {
  
	private static volatile ImageRectification imagerectify = null;
	private static volatile Auto_LF_Deconvolution auto_LF = null;
	
	public static void main(final String... args) throws Exception {
		
	
		String fonts = "Serif";

		JFrame.setDefaultLookAndFeelDecorated(true);
		JFrame frame = new JFrame("Auto_LF_Deconvolution");
		frame.setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
		frame.setSize(385, 200);
		frame.setLocationRelativeTo(null);
		frame.setResizable(true);
		
		JPanel panel01 = new JPanel();

		panel01.setLayout(null);
		frame.add(panel01);
		
		JButton isOK_Button = new JButton("ImageRectify");
		isOK_Button.setForeground(Color.BLUE);
		isOK_Button.setFont(new java.awt.Font(fonts, 0, 28));
		isOK_Button.setBounds(10, 30, 350, 35);
		panel01.add(isOK_Button);
		isOK_Button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				imagerectify = new ImageRectification();
			}
		});

		JButton isStop_Button = new JButton("Auto_LF_Deconvolution");
		isStop_Button.setForeground(Color.BLUE);
		isStop_Button.setFont(new java.awt.Font(fonts, 0, 28));
		isStop_Button.setBounds(10, 100, 350, 35);
		panel01.add(isStop_Button);
		isStop_Button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				auto_LF = new Auto_LF_Deconvolution();

			}
		});
		frame.setVisible(true);
			
		while(true) {
			if(imagerectify!=null) {
				imagerectify.run(null);
				imagerectify = null;
			}
			if(auto_LF!=null) {
				frame.setVisible(false);
				auto_LF.run(null);
				auto_LF = null;
			}
		}		
	}	
}
