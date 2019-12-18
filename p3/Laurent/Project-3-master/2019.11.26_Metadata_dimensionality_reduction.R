Env_data=read.table("C:/Users/laurenfo/Documents/COMSAT/COMSAT_environmental_metadata(spectral_data_lakes).tsv", stringsAsFactors=FALSE, header=T)
#Env.mat=scale(Env_data[,c(3:41, 43:44)], scale=T, center=T)
Env.mat=Env_data[,c(3:41, 43:44)]
#rownames(Env.mat) = rownames.OTU.matching.spectra
Taxonomy.table = readRDS("C:/Users/laurenfo/Documents/COMSAT/2019.01.31_Output_AE/tax_final.rds")	#Load Taxonomy table.
OTU.table = readRDS("C:/Users/laurenfo/Documents/COMSAT/2019.01.31_Output_AE/seqtab_nochim.rds")	#Load OTU table.
Spectra.table = read.table("C:/Users/laurenfo/Documents/COMSAT/Absorption spectra/COMSAT 2011 Absorption spectra DOM.txt", header=T, check.names=F)

rownames(OTU.table) = c(							#Change sequencing run IDs for lake ID and name in the OTU table.
"10000_Hurdalsjøen"  ,   "10001_Harestuvatnet"   , 
"170B_Gjersjøen"     ,   "170_Gjersjøen"         ,
"180_Øgderen"        ,   "189_Krøderen"          ,
"191_Rødbyvatnet"    ,  "194_Sperillen"          ,
"214_Gjesåssjøen"    , "2252_Rotnessjøen"        ,
"2268_Mylla"         ,     "233_Osensjøen"       ,   
"236_Rokossjøen"     ,   "2374_Klämmingen"       ,
"242_Sør Mesna"      ,   "252_Vermundsjøen"      ,
"261_Kalandsvatnet"  ,     "264_Myrkdalsvatnet"  ,   
"2678_Torrsjøn"      ,   "285_Rotevatnet"        ,
"2870_Visten"        ,     "2875_Näsrämmen"      ,
"2878_Rangsjön"      ,   "2887_Tisjön"           ,
"2888_Halsjøen"      ,   "288_Vatnevatnet"       , 
"2899_Jangen"        ,     "3017_Sör-älgen"      ,
"3019_Möckeln"       ,   "3020_Ljusnaren"        , 
"3025_Halvarsnoren"  ,     "3027_Nätsjön"        ,
"3029_Örlingen"      ,   "3031_Saxen"            , 
"3106_Långbjörken"   , "3160_Skattungen"         ,
"3165_Bäsingen"      ,   "3167_Tisken"           , 
"3185_Stora Almsjön" ,  "3189_Dragsjön"          ,
"3201_Milsjön"       ,   "3220_Stora Korslängen" ,
"326_Einavatnet"     ,     "328_Randsfjorden"    ,   
"3384_Hinsen"        ,    "3397_Storsjön"        ,
"3399_Grycken"       ,     "339_Ringsjøen"       ,
"340_Sæbufjorden"    ,   "344_Strondafjorden"    ,
"345_Trevatna"       ,    "349_Bogstadvannet"    ,
"3516_Holmsjön"      ,   "353_Aspern"            ,
"3541_Stornaggen"    ,     "361_Rødenessjøen"    ,
"363_Rømsjøen"       , "378_Hetlandsvatn"        ,
"380_Lutsivatn"      ,     "394_Vatsvatnet"      ,   
"395_Vostervatnet"   ,     "404_Jølstravatnet"   ,
"405_Oppstrynvatnet" ,     "433_Bandak"          , 
"436_Grungevatnet"   ,     "453_Vinjevatn"       ,
"481_Åsrumvatnet"    ,   "482_Bergsvannet"       ,
"486B_Goksjø"         ,  "486_Goksjø"            ,
"487_Hallevatnet"    ,     "498_Dagarn"          ,    
"5000_Forsjösjön"    , "519_Langen"
)

Spectra.table.2 = read.table("C:/Users/laurenfo/Documents/COMSAT/Absorption spectra/COMSAT 2011 Absorption spectra DOM ordered for 16S data.txt", header=T, check.names=F)
Spectra.mat = t(Spectra.table.2[,2:73])		#Select spectral data sites matching 16S data sites.
colnames(Spectra.mat) = Spectra.table.2[,1]
#Sample 353 from the OTU table has no corresponding spectral data.
OTU.table.spectral = OTU.table[c(1:7, 9:53, 55:74),]
OTU.table.spectral = OTU.table.spectral[,colSums(OTU.table.spectral)>0]	#Remove OTUs absent from site subset.
Taxonomy.table.spectral = Taxonomy.table[colnames(OTU.table.spectral),] #Remove OTUs absent from site subset.

rownames.OTU.matching.spectra = c(
"10000_Hurdalsjøen"  ,   "10001_Harestuvatnet"   , 
"170B_Gjersjøen"     ,   "170_Gjersjøen"         ,
"180_Øgderen"        ,   "189_Krøderen"          ,
"191_Rødbyvatnet"    ,
"214_Gjesåssjøen"    , "2252_Rotnessjøen"        ,
"2268_Mylla"         ,     "233_Osensjøen"       ,   
"236_Rokossjøen"     ,   "2374_Klämmingen"       ,
"242_Sør Mesna"      ,   "252_Vermundsjøen"      ,
"261_Kalandsvatnet"  ,     "264_Myrkdalsvatnet"  ,   
"2678_Torrsjøn"      ,   "285_Rotevatnet"        ,
"2870_Visten"        ,     "2875_Näsrämmen"      ,
"2878_Rangsjön"      ,   "2887_Tisjön"           ,
"2888_Halsjøen"      ,   "288_Vatnevatnet"       , 
"2899_Jangen"        ,     "3017_Sör-älgen"      ,
"3019_Möckeln"       ,   "3020_Ljusnaren"        , 
"3025_Halvarsnoren"  ,     "3027_Nätsjön"        ,
"3029_Örlingen"      ,   "3031_Saxen"            , 
"3106_Långbjörken"   , "3160_Skattungen"         ,
"3165_Bäsingen"      ,   "3167_Tisken"           , 
"3185_Stora Almsjön" ,  "3189_Dragsjön"          ,
"3201_Milsjön"       ,   "3220_Stora Korslängen" ,
"326_Einavatnet"     ,     "328_Randsfjorden"    ,   
"3384_Hinsen"        ,    "3397_Storsjön"        ,
"3399_Grycken"       ,     "339_Ringsjøen"       ,
"340_Sæbufjorden"    ,   "344_Strondafjorden"    ,
"345_Trevatna"       ,    "349_Bogstadvannet"    ,
"3516_Holmsjön"      ,
"3541_Stornaggen"    ,     "361_Rødenessjøen"    ,
"363_Rømsjøen"       , "378_Hetlandsvatn"        ,
"380_Lutsivatn"      ,     "394_Vatsvatnet"      ,   
"395_Vostervatnet"   ,     "404_Jølstravatnet"   ,
"405_Oppstrynvatnet" ,     "433_Bandak"          , 
"436_Grungevatnet"   ,     "453_Vinjevatn"       ,
"481_Åsrumvatnet"    ,   "482_Bergsvannet"       ,
"486B_Goksjø"         ,  "486_Goksjø"            ,
"487_Hallevatnet"    ,     "498_Dagarn"          ,    
"5000_Forsjösjön"    , "519_Langen"
)

rownames(Spectra.mat) = rownames.OTU.matching.spectra

#Interpolate missing values in environmental data by Multivariate Imputations by Chained Equations (MICE)
library(mice)
Env.mat.interpolated.NA=complete(mice(Env.mat))
rownames(Env.mat.interpolated.NA) = rownames.OTU.matching.spectra

Env.scaled = scale(Env.mat.interpolated.NA, scale=T, center=T)	#Scaled metadata table

reorder_cor.mat = function(cor.mat){
	# Use correlation between variables as distance
	dd = as.dist((1-cor.mat)/2)
	hc = hclust(dd)
	cor.mat = cor.mat[hc$order, hc$order]
	return(cor.mat)
	}

library(reshape2)

get_lower_triangle = function(cor.mat){
    cor.mat[upper.tri(cor.mat)] <- NA
    return(cor.mat)
  }
get_upper_triangle = function(cor.mat){
    cor.mat[lower.tri(cor.mat)]<- NA
    return(cor.mat)
  }

cor_p = function(env.mat){
	env.cor = melt(get_lower_triangle(reorder_cor.mat(cor(env.mat, use="all.obs", method="pearson"))), na.rm = TRUE)
	env.cor.p = env.cor
	for (i in 1:nrow(env.cor)){
		v1 = as.character(env.cor[i, 1])
		v2 = as.character(env.cor[i, 2])
		env.cor.p[i,3] = p.adjust(cor.test(env.mat[,v1],env.mat[,v2])$p.value, method = "holm", n = length(env.cor$value))
		}
	return(list(env.cor, env.cor.p))
	}

env.mat.list = cor_p(Env.scaled)
env.cor = env.mat.list[[1]]
env.cor.p = env.mat.list[[2]]

# Heatmap
library(ggplot2)
pdf("Environmental_variables_correlation_matrix.pdf")
ggplot(data = env.cor, aes(Var2, Var1, fill = value))+
 geom_tile(color = "white")+
 scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
   midpoint = 0, limit = c(-1,1), space = "Lab", 
   name="Pearson\nCorrelation") +
  theme_minimal()+ 
 scale_x_discrete(position = "top") +
 theme(axis.text.x = element_text(angle = 45, vjust = 1, 
    size = 8, hjust = 0))+
 coord_fixed()+
 # P values for correlations are shown as text in tiles
 geom_text(data = env.cor.p, aes(Var2, Var1, label = round(value, 2)), color = "black", size = 1.5) +
theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  panel.grid.major = element_blank(),
  panel.border = element_blank(),
  panel.background = element_blank(),
  axis.ticks = element_blank(),
  legend.justification = c(1, 0),
  legend.position = c(0.6, 0.1),
  legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                title.position = "top", title.hjust = 0.5))
dev.off()

env.cor.p[env.cor.p$Var1=="a.dom.m"|env.cor.p$Var2=="a.dom.m",]	#p-values for correlations with CDOM
env.cor.p[env.cor.p$Var1=="a.dom.m" & env.cor.p$value<0.05|env.cor.p$Var2=="a.dom.m" & env.cor.p$value<0.05,]	#p-values for significant correlations with CDOM
env.cor.p[env.cor.p$value<0.05 & env.cor.p$Var1!=env.cor.p$Var2,]	#Correlations and p-values for significantly correlated variables

network_df = function(env.mat, p.value = 0.05){
	env.cor.out = cor_p(env.mat)
	env.cor = env.cor.out[[1]]
	env.cor.p = env.cor.out[[2]]
	raw.df = env.cor.p[env.cor.p$value<p.value & env.cor.p$Var1!=env.cor.p$Var2,]
	node.list = as.data.frame(cbind(matrix(1:length(colnames(env.mat)),length(colnames(env.mat)),1), matrix(colnames(env.mat),length(colnames(env.mat)),1)))
	node.list$V1 = 1:nrow(node.list)
	node.list$V2 = as.character(node.list$V2)
	network.p.df = raw.df
	network.p.df$Var1 = 1:nrow(raw.df)
	network.p.df$Var2 = 1:nrow(raw.df)
	network.cor.df = network.p.df
	for (i in 1:nrow(raw.df)){
		network.p.df[i,1] = as.integer(node.list[node.list$V2==as.character(raw.df[i, 1]), 1])
		network.p.df[i,2] = as.integer(node.list[node.list$V2==as.character(raw.df[i, 2]), 1])
		network.p.df[i,3] = as.numeric(-log(raw.df[i, 3]))
		network.cor.df[i,1] = network.p.df[i,1]
		network.cor.df[i,2] = network.p.df[i,2]
		network.cor.df[i,3] = abs(env.cor[env.cor$Var1==as.character(raw.df$Var1[i]) & env.cor$Var2==as.character(raw.df$Var2[i]), 3])
		}
	return(list(network.cor.df, network.p.df, node.list))
	}

library(network)
library(sna)
library(ggplot2)
library(GGally)
pdf("Metadata_networks.pdf")
network_df.out = network_df(Env.scaled, 0.05)
env.network_cor_df = network_df.out[[1]]
env.network_p_df = network_df.out[[2]]
env.node_list = network_df.out[[3]]
env.network = network(env.network_cor_df, vertex.attr = env.node_list, matrix.type = "edgelist", directed = FALSE)
ggnet2(env.network, size = "degree", label=env.node_list$V2, label.size=2, edge.label.size = 0.1, edge.size = 0.01)+
ggtitle(label="p=0.05")

network_df.out = network_df(Env.scaled, 5e-4)
env.network_cor_df = network_df.out[[1]]
env.network_p_df = network_df.out[[2]]
env.node_list = network_df.out[[3]]
env.network = network(env.network_cor_df, vertex.attr = env.node_list, matrix.type = "edgelist", directed = FALSE)
ggnet2(env.network, size = "degree", label=env.node_list$V2, label.size=2, edge.label.size = 0.1, edge.size = 0.01)+
ggtitle(label="p=0.0005")
dev.off()