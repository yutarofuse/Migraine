rm( list = ls( envir = globalenv() ), envir = globalenv() )

Internal <- read_csv("Internal_dataset.csv")
External <- read_csv("External_dataset.csv")

Internal <- Internal %>% mutate("HOSPITAL" = 0)
External <- External %>% mutate("HOSPITAL" = 1)

colnames(Internal) <- gsub("_internal", "_", colnames(Internal))

colnames(External) <- gsub("_external", "_", colnames(Internal))

ALL2 <- rbind(Internal,External)

ALL3 <- ALL2 %>% select(-Date, -patient, -Date_)

library(gtsummary)
library(exactRankTests)
library(flextable)

Labtable<- ALL3

LabdataTABLE <- 
  Labtable %>% 
  mutate(HOSPITAL = factor(HOSPITAL, labels = c("1", "2"))) %>% 
  tbl_summary(by = HOSPITAL, statistic = list(all_continuous() ~ "{median}({p25}, {p75})", Age ~ "{mean} ({sd})"))%>% 
  add_p(list(all_continuous() ~ "wilcox.test",
             all_categorical() ~ "fisher.test"))  %>% 
  bold_p(t= 0.05) %>% 
  add_overall() %>% 
  modify_spanning_header(c("stat_1", "stat_2") ~ "**Hospital**") %>% 
  modify_header(label = "**Input variables**") %>% 
  as_flex_table() %>% 
  save_as_docx(path ="HospitalDiff2.docx")

LabdataTABLE