/*
 * /*******************************************************************************
 *  * Copyright (c) 2012  DCA-FEEC-UNICAMP
 *  * All rights reserved. This program and the accompanying materials
 *  * are made available under the terms of the GNU Lesser Public License v3
 *  * which accompanies this distribution, and is available at
 *  * http://www.gnu.org/licenses/lgpl.html
 *  * 
 *  * Contributors:
 *  *     K. Raizer, A. L. O. Paraense, R. R. Gudwin - initial API and implementation
 *  ******************************************************************************/
 
package cst_procedural_app;

import outsideCommunication.OutsideCommunication;

import java.io.File;
import java.io.IOException;
/**
 *
 * 
 * @author L. L. Rossi (leolellisr)
 */
public class CST_procedural {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
    	// removing previous .txt files expect QTable
    	File folder = new File(".");
    	for (File f : folder.listFiles()) {
    		if(f.getName().endsWith(".txt") && !(f.getName().endsWith("QTable.txt"))) {
    			f.delete();
    		}
    	}
        Boolean sensorialTest = false;
        Boolean attentionalTest = false;
        Boolean printMaps = true;
        if(sensorialTest || attentionalTest)  printMaps = false;
        
        String mode = "learning";
        String model = "dqn";
        int stage = 3, exp =1, res = 256, 
                max_time_graph=100, MAX_ACTION_NUMBER = 500;
        long seed = 1234;
        
        int n_tables = 1;
        String runId=""; 
        int num_pioneer = 1, num_episodes = 50;
        
        OutsideCommunication oc = new OutsideCommunication(num_episodes,mode,n_tables,seed, stage, 
                exp, "", res, max_time_graph, MAX_ACTION_NUMBER, num_pioneer, printMaps, attentionalTest);
        oc.start(); 
        //  (OutsideCommunication oc, String mode, String motivation, int num_tables, int print_step)
        AgentMind am = new AgentMind(oc, mode, "drives",n_tables, 5,seed, 
                num_pioneer,model,sensorialTest, attentionalTest, printMaps); // OC, mode, Num_QTables,  PrintStep, seed, num_pioneer, 

    }
    
}
