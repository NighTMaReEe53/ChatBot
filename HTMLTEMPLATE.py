
css ='''
  <style>

  body {
        font-family: Tajawal, system-ui;
  }

  .st-emotion-cache-13k62yr, .st-emotion-cache-gquqoo {
    background-color: #030303;
  }

  .st-emotion-cache-1k9kca4, .st-emotion-cache-zh4rd8 {
    background-color: rgb(255 255 255 / 4%)
  }

  ._terminalButton_rix23_138 {
    display: none;
  }

  .st-emotion-cache-zh4rd8 {
    transition: 0.3s ease;
  }

  .st-emotion-cache-1iitq1e {
        width: 100%;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 1rem;
    border-radius: 0.5rem;
    background: rgba(255, 255, 255, 0.02);
    direction: rtl;
    font-family: Tajawal;
    box-shadow: 0 15px 60px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
  }


  h2, h3, h1, h5, h6, h4, p, span {
    font-family: Tajawal, system-ui;
  }

  p {
    font-size: 18px;
  }

  .st-emotion-cache-z68l0b {
  
    color: #F2f2f2;
    border-radius: 6px;
    background: rgba(53, 245, 160, 0.12);
    border: 1px solid rgba(53, 245, 160, 0.25);
    box-shadow: 0 8px 25px rgb(22 65 45);
  }

  
  [data-testid="stChatMessage"][data-role="assistant"] p {
  color: #148b54;
  }

  [data-testid="stChatMessage"][data-role="assistant"] 
[data-testid="stMarkdownContainer"] {
    color: #148b54;
}

  
      .st-emotion-cache-khw9fs {
    color: #f2f2f2;
    background: rgb(251 0 0 / 27%);
    border: 1px solid rgb(233 30 99 / 15%);
    box-shadow: 0 8px 25px rgb(255 0 0 / 15%);
  }



  .st-emotion-cache-1w4gzkv h1 {
    text-align: center;
    font-family: Tajawal, system-ui
  }

  .st-emotion-cache-9rsxm2 p {
    font-size: 15px;
    font-family: Tajawal;
    font-weight: bold;
    text-align: center
  }

  .st-emotion-cache-9rsxm2 {
    width:100%
  }

  ol, ul {
    padding-right: 15px;
    font-family: Tajawal;
  }
  .st-emotion-cache-467cry h1, .st-emotion-cache-467cry h2, .st-emotion-cache-467cry h3, .st-emotion-cache-467cry h4, .st-emotion-cache-467cry h5, .st-emotion-cache-467cry h6 {
    font-family: Tajawal;
  }

  .st-ao {
    background-color: #FFF1;
  }
  .st-as {
    background-color: #ffc107;
  }

  .overlay {
    position: fixed;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background-color: #FFF;
    top: 0px;
    left: 0px;
    z-index: 99999999999999999;
    filter: blur(200px);
    pointer-events: none;
    }

    .st-emotion-cache-6shykm {
      padding:20px
    }

    .st-emotion-cache-hzygls {
      background-color: #0003;
      backdrop-filter: blur(10px);
    }

    .st-emotion-cache-x1bvup {
      background-color: #000;
      border: 1px solid #fff1;
      font-family: Tajawal;
      transition: 0.3s ease;
      &:focus {
        border: 1px solid #009688;
      }
    }

        .st-emotion-cache-x1bvup textarea {
          background-color: #000;
        }

        .st-emotion-cache-x1bvup > div:focus {
          border: 2px solid green;
          outline: 2px solid green;
        }

/* 🔹 تنسيق الجداول داخل الردود */
div[data-testid="stMarkdownContainer"] table {
    border-collapse: collapse;
    width: 100%;
    max-width: 100%;
    display: block;
    overflow-x: auto;
    border-radius: 10px;
    margin-top: 8px;
}

/* 🔹 جعل الجدول يمكن تمريره أفقياً داخل الصندوق */
div[data-testid="stMarkdownContainer"] table thead {
    background: rgba(255, 255, 255, 0.08);
}
div[data-testid="stMarkdownContainer"] table th,
div[data-testid="stMarkdownContainer"] table td {
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 8px 12px;
    color: #fff;
    font-family: "Tajawal", sans-serif;
    font-size: 15px;
    text-align: center;
    word-break: break-word;
}

/* 🔹 تمرير الجداول أفقيًا فقط داخل الصندوق */
.response-box {
    overflow-x: auto;
    max-width: 100%;
    white-space: nowrap;
}

/* 🔹 تحسين عرض الجداول على الموبايل */
@media (max-width: 768px) {
    div[data-testid="stMarkdownContainer"] table {
        font-size: 14px;
    }
    div[data-testid="stMarkdownContainer"] table th,
    div[data-testid="stMarkdownContainer"] table td {
        padding: 6px 8px;
    }
}

@media(max-width:776px) {

.st-emotion-cache-467cry h1, .st-emotion-cache-467cry h2, .st-emotion-cache-467cry h3, .st-emotion-cache-467cry h4, .st-emotion-cache-467cry h5, .st-emotion-cache-467cry h6 {

  font-size: 30px;
}

}
  .st-emotion-cache-hkjmcg {
    background-color: #000;
  }

  textarea {
    padding-right : calc(1rem) !important;
    direction: rtl;
  }

  .st-emotion-cache-sey4o0 {
    left: 0px;
  }

  .st-emotion-cache-1jooqly:hover {
    color: #8BC34A;
  }

  .st-emotion-cache-x1bvup {
    transition: 0.3s ease;
  }

  .st-emotion-cache-x1bvup:focus-within {
    border-color: #8BC34A;
  }
  


    @media(max-width:992px) {
        .st-emotion-cache-1p2n2i4 {
    bottom: -10px;
  }

    }


    @media(max-width:992px) {
        .st-emotion-cache-tn0cau {
    margin-bottom: 50px
  }

    }

    .st-emotion-cache-1mph9ef {
      background-color: rgba(255, 255, 255, 0.06);
      margin: 15px 0px 0px 0px;
      direction: rtl;
      box-shadow: 0 15px 60px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(16px);
    }

  </style>

'''

